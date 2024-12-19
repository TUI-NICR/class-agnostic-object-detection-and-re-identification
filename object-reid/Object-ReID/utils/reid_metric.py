# encoding: utf-8

import numpy as np
import torch
import pandas as pd
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func_me
from survey.data.datasets.eval_reid import eval_func


class r1_mAP_mINP(Metric):
    """
    Metric used for object ReiD.

    Args:
    - num_query (int): Number of queries in dataset
    - max_rank (int): Doesn't do anything. Bug in original implementation?
    - feat_norm (str): "on" or "off". Whether features should be normalized
    - save_memory (str): IMPORTANT! "on" or "off". Whether to use the metric implemented for object ReiD. Should be "on"!
    - block_size (int): Size of chunk of distance matrix to be calculated at once. Higher means faster but more memory usage
    - logger (Logger): Logger to track metric calculation progress
    """
    def __init__(self, num_query, max_rank=50, feat_norm='on', save_memory='off', block_size=2000, logger=None):
        super(r1_mAP_mINP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.save_memory = save_memory
        self.block_size = block_size
        self.logger = logger

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.pclasses = []

    def update(self, output):
        feat, pid, camid, other = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.pclasses.extend(np.asarray(other[0]))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'on':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_classes = np.asarray(self.pclasses[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_classes = np.asarray(self.pclasses[self.num_query:])
        tabular_data = None
        if self.save_memory == 'on':
            cmc, mAP, mINP, tabular_data = eval_func_me(qf, gf, q_pids, g_pids, q_camids, g_camids, q_classes, g_classes, block_size=self.block_size, logger=self.logger)
            tabular_data = pd.DataFrame.from_dict({
                "id": q_pids,
                "class": q_classes,
                "avg_dist_match": tabular_data[0],
                "Q25_dist_match": tabular_data[1],
                "Q50_dist_match": tabular_data[2],
                "Q75_dist_match": tabular_data[3],
                "avg_dist_mismatch_class": tabular_data[4],
                "avg_dist_mismatch": tabular_data[5],
                "Q25_dist_mismatch": tabular_data[6],
                "Q50_dist_mismatch": tabular_data[7],
                "Q75_dist_mismatch": tabular_data[8],
                "first_miss": tabular_data[9],
                "last_hit": tabular_data[10],
                "AP": tabular_data[11]
            })
        else:
            m, n = qf.shape[0], gf.shape[0]
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.cpu().numpy()
            cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        del feats, qf, gf
        torch.cuda.empty_cache()
        return cmc, mAP, mINP, tabular_data
