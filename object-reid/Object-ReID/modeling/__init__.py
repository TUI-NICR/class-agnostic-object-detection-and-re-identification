# encoding: utf-8

from .baseline import Baseline


def build_model(cfg, num_classes, num_superclasses=None):
    model = Baseline(
        num_classes,
        num_superclasses,
        cfg.MODEL.LAST_STRIDE,
        cfg.MODEL.PRETRAIN_PATH,
        cfg.MODEL.NAME,
        cfg.MODEL.GENERALIZED_MEAN_POOL,
        cfg.MODEL.PRETRAIN_CHOICE,
        cfg.MODEL.REDUCE_DIM,
        cfg.MODEL.REDUCED_DIM,
        cfg.MODEL.HIERARCHICAL_CLS
    )
    return model
