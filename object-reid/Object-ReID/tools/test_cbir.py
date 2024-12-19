#!/usr/bin/env python
#SBATCH --gres=gpu:1
#SBATCH -p long
#SBATCH -t 1:00:00
#SBATCH --exclude=jupiter2,titan,titan2
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=henning.franke@tu-ilmenau.de

# encoding: utf-8

"""
Scripts used to perform evaluation of a CBIR model on a ReID dataset.
Cobbled together from ReID-Survey and SuperGlobal Code.
"""

import argparse
import sys
import os
import torch
import cv2

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from yacs.config import CfgNode as CN

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append('.')
sys.path.append(cwd + "/../")
from data import init_dataset

from SuperGlobal import CVNet_Rerank, load_checkpoint, RerankwMDA, MDescAug

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEPTH = 50
_C.MODEL.REDUCTION_DIM = 2048

_C.TEST = CN()
_C.TEST.WEIGHTS = ""
_C.TEST.DATA_DIR = ""
_C.TEST.DATASET = ""
_C.TEST.SCALE_LIST = 3

_C.SupG = CN()
_C.SupG.relup = True
_C.SupG.gemp = True
_C.SupG.rgem = True
_C.SupG.sgem = True
_C.SupG.rerank = True

cfg = _C

_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]


def eval_func(indices, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = indices.shape
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP


class DataSet(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, dataset, split):
        if split == "query":
            self._dataset = dataset.query
        else:
            self._dataset = dataset.gallery
        self._split = split
        self._ds_type = dataset.ds_type

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        for i in range(im.shape[0]):
            im[i] = im[i] - _MEAN[i]
            im[i] = im[i] / _SD[i]
        im = torch.tensor(im)
        return im

    def __getitem__(self, index):
        # Load the image
        if self._ds_type == "object":
            img_path, pid, camid, obj_class, img_num, ymax, ymin, xmax, xmin = self._dataset[index]
            im = cv2.imread(img_path)
        elif self._ds_type == "object-masked":
            img_path, img_mask_path, pid, camid, obj_class, img_num, ymax, ymin, xmax, xmin = self._dataset[index]
            im = cv2.imread(img_path)
            mask = cv2.imread(img_mask_path, cv2.IMREAD_GRAYSCALE)
            im[mask == 0] = [0, 0, 0]
        im = im[ymin:ymax, xmin:xmax]
        im = im.astype(np.float32, copy=False)
        im = self._prepare_im(im)
        return im, pid, camid, obj_class

    def __len__(self):
        return len(self._dataset)


def collate_fn(batch):
    imgs, pids, camids, obj_classs = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, obj_classs


@torch.no_grad()
def extract_feature(model, data_dir, dataset, split, gemp, rgem, sgem, scale_list):
    with torch.no_grad():
        dataset = init_dataset(dataset, root=data_dir)
        cbir_dataset = DataSet(dataset, split)
        # Create a loader
        test_loader = torch.utils.data.DataLoader(
            cbir_dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            num_workers=4,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn
        )
        img_feats = []
        pids = []
        camids = []
        classes = []

        for im, pid, camid, obj_class in tqdm(test_loader):
            im = im.cuda()

            desc = model.extract_global_descriptor(im, gemp, rgem, sgem, scale_list)

            if len(desc.shape) == 1:
                desc.unsqueeze_(0)
            img_feats.append(desc)
            pids += pid
            camids += camid
            classes += obj_class

        img_feats = torch.cat(img_feats, dim=0)
        img_feats = F.normalize(img_feats, p=2, dim=1)

    return img_feats, pids, camids, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="./configs/CBIR/cbir_cfg.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = CVNet_Rerank(cfg.MODEL.DEPTH, cfg.MODEL.REDUCTION_DIM, cfg.SupG.relup)
    model = model.to(device="cuda")

    load_checkpoint(cfg.TEST.WEIGHTS, model)

    model.eval()
    state_dict = model.state_dict()

    MDescAug_obj = MDescAug()
    RerankwMDA_obj = RerankwMDA()

    model.load_state_dict(state_dict)
    Q, Q_pids, Q_camids, Q_classes = extract_feature(model, cfg.TEST.DATA_DIR, cfg.TEST.DATASET, "query", cfg.SupG.gemp, cfg.SupG.rgem, cfg.SupG.sgem, cfg.TEST.SCALE_LIST)
    X, X_pids, X_camids, X_classes = extract_feature(model, cfg.TEST.DATA_DIR, cfg.TEST.DATASET, "db", cfg.SupG.gemp, cfg.SupG.rgem, cfg.SupG.sgem, cfg.TEST.SCALE_LIST)

    sim = torch.matmul(X, Q.T)
    ranks = torch.argsort(-sim, axis=0)
    if cfg.SupG.rerank:
        rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = MDescAug_obj(X, Q, ranks)
        ranks = RerankwMDA_obj(ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)
    ranks = ranks.data.cpu().numpy().T
    Q_pids = np.array(Q_pids)
    X_pids = np.array(X_pids)
    Q_camids = np.array(Q_camids)
    X_camids = np.array(X_camids)
    all_cmc, mAP, mINP = eval_func(ranks, Q_pids, X_pids, Q_camids, X_camids)
    print(mAP)


if __name__ == "__main__":
    main()
