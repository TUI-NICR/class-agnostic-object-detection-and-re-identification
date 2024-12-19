# encoding: utf-8
import logging
import numpy as np
import json
import torch
from ignite.engine import Engine
from ignite.metrics import Metric


class ImgSaver(Metric):
    """
    Hook which takes model outputs, performs inference and returns the results
    as a dictionary.
    - num_query (int): Number of queries in dataset
    - num_retrieve (int): Number of ranking results that should be saved for each query
    - feat_norm (str): "on" or "off". Whether to normalize the features
    """
    def __init__(self, num_query, num_retrieve, feat_norm='on'):
        super(ImgSaver, self).__init__()
        self.num_query = num_query
        self.num_retrieve = num_retrieve
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.img_paths = []
        self.pids = []
        self.classes = []

    def update(self, output):
        feat, img_path, pid, other = output
        self.feats.append(feat)
        self.img_paths.extend(np.asarray(img_path))
        self.pids.extend(np.asarray(pid))
        self.classes.extend(np.asarray(other[0]))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'on':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_paths = np.asarray(self.img_paths[:self.num_query])
        q_pids = np.asarray(self.pids[:self.num_query])
        q_classes = np.asarray(self.classes[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_paths = np.asarray(self.img_paths[self.num_query:])
        g_pids = np.asarray(self.pids[self.num_query:])
        g_classes = np.asarray(self.classes[self.num_query:])

        m, n = qf.shape[0], gf.shape[0]
        matches = {}
        for q_idx, q in enumerate(qf):
            distmat = torch.pow(q, 2).sum().unsqueeze(dim=0).expand(1, n) + \
                    torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, 1).t()
            distmat.addmm_(1, -2, q.unsqueeze(dim=0), gf.t()).squeeze()
            distmat = distmat.cpu().numpy()
            indices = np.argsort(distmat)
            matches[q_paths[q_idx]] = {
                "id": int(q_pids[q_idx]),
                "class": q_classes[q_idx],
                "results": [
                    {"id": int(i), "class": c, "path": p, "distance": float(d)} for i, c, p, d in zip(
                        g_pids[indices[0, :self.num_retrieve]].tolist(),
                        g_classes[indices[0, :self.num_retrieve]].tolist(),
                        g_paths[indices[0, :self.num_retrieve]].tolist(),
                        distmat[0, indices[0, :self.num_retrieve]].tolist()
                    )
                ]
            }
        return matches


def create_supervised_inferencer(model, num_query, num_retrieve, feat_norm, device=None):
    """
    Factory function for creating an inferencer for supervised models

    Args:
        model (`torch.nn.Module`): the model to evaluate
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an inferencer engine with supervised inference function
    """

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, img_paths, pids, other = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, img_paths, pids, other

    engine = Engine(_inference)
    img_save = ImgSaver(num_query, num_retrieve, feat_norm)
    img_save.attach(engine, "img_save")

    return engine


def do_inference(
        cfg,
        model,
        data_loader,
        num_inference_samples
):
    """
    Do inference on dataset and save results into a .json file.

    Args:
    - cfg (CfgNode): Config of experiment
    - model (Module): Loaded torch model
    - data_loader (dict): Dict containing DataLoader at key "inference"
    - num_inference_samples (int): Number of inference samples to retrieve. Must match number of queries in dataset of data_loader!
    """
    device = cfg.MODEL.DEVICE
    logger = logging.getLogger("reid_baseline")
    logger.info("Enter inferencing")
    inferencer = create_supervised_inferencer(
        model,
        num_query=num_inference_samples,
        num_retrieve=cfg.INFERENCE.NUM_RETRIEVE,
        feat_norm=cfg.INFERENCE.FEAT_NORM,
        device=device
    )

    inferencer.run(data_loader['inference'])
    matches = inferencer.state.metrics['img_save']
    for q, ms in matches.items():
        logger.info(f'Results for query {q}: {ms}')
    with open(cfg.INFERENCE.OUTPUT, "w") as f:
        json.dump(matches, f)
