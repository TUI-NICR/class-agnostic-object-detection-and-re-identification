"""
This is a simplified evaluation script which applies a model to a dataset
and saves the calculated distance matrix into a Pandas dataframe. The entire
matrix is calculated in one go so this only works with smaller datasets.
"""
import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.backends import cudnn

sys.path.append(os.getcwd())
sys.path.append('.')
from config import cfg
from data import make_data_loader
from modeling import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="./configs/Embedding_Analysis/CO3D_v1_transfer.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = cfg.MODEL.DEVICE
    if device == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True

    data_loader, num_query, num_classes, num_inference_samples, num_superclasses = make_data_loader(cfg)
    model = build_model(cfg, num_classes)

    if 'cpu' not in device:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device=device)

    model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    results = []
    for batch in tqdm(data_loader['eval']):
        with torch.no_grad():
            data, pids, camids, other = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            results.append((feat, np.asarray(pids), np.asarray(other[0])))
    results = list(zip(*results))
    feats = torch.cat(results[0], dim=0)
    if cfg.TEST.FEAT_NORM == 'on':
        print("The test feature is normalized")
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    pids = np.concatenate(results[1])
    pclasses = np.concatenate(results[2])
    # query
    qf = feats[:num_query]
    q_pids = pids[:num_query]
    q_classes = pclasses[:num_query]
    # gallery
    gf = feats[num_query:]
    g_pids = pids[num_query:]
    g_classes = pclasses[num_query:]

    m, n = qf.shape[0], gf.shape[0]
    distmat_ = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat_.addmm_(1, -2, qf, gf.t())
    distmat = distmat_.cpu().numpy()
    del distmat_
    torch.cuda.empty_cache()
    df = pd.DataFrame(distmat, index=[q_classes, q_pids], columns=[g_classes, g_pids])
    name = cfg.TEST.WEIGHT.split("/log/")[1].split(".pt")[0].replace("/", "__") + "__" + cfg.DATASETS.NAMES + ".pkl"
    df.to_pickle(os.path.join(cfg.OUTPUT_DIR, name))


if __name__ == "__main__":
    main()
