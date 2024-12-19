# encoding: utf-8

import torch
from torch.utils.data import DataLoader
import numpy as np

from .datasets import init_dataset, ImageDataset
from .triplet_sampler import RandomClassSampler, ClassBalancedSampler, RandomIdentitySampler
from .transforms import build_transforms, BatchPad


def _pad(imgs, fill):
    xmax = max([img.shape[-2] for img in imgs])
    ymax = max([img.shape[-1] for img in imgs])
    padding = BatchPad((xmax, ymax), fill)
    return [padding(img) for img in imgs]


def train_collate_fn(class_map: dict, keep_ratio: bool, fill=0):
    def collate_fn(batch):
        imgs, pids, _, _, other = zip(*batch)
        other = list(zip(*other))
        if class_map is not None:
            other[0] = torch.tensor([class_map[e] for e in other[0]], dtype=torch.int64)
        if keep_ratio:
            imgs = _pad(imgs, fill)
        pids = torch.tensor(pids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, other
    return collate_fn


def val_collate_fn(keep_ratio: bool, fill=0):
    def collate_fn(batch):
        imgs, pids, camids, _, other = zip(*batch)
        other = list(zip(*other))
        if keep_ratio:
            imgs = _pad(imgs, fill)
        return torch.stack(imgs, dim=0), pids, camids, other
    return collate_fn


def inference_collate_fn(keep_ratio: bool, fill=0):
    def collate_fn(batch):
        imgs, pids, _, img_paths, other = zip(*batch)
        other = list(zip(*other))
        if keep_ratio:
            imgs = _pad(imgs, fill)
        return torch.stack(imgs, dim=0), img_paths, pids, other
    return collate_fn


def make_data_loader(cfg):
    """
    Create dataloader from config.

    Args:
    - cfg (CfgNode): config for experiment.
    """
    transforms = build_transforms(cfg)
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    ds_type = dataset.ds_type
    # number od IDs
    num_classes = dataset.num_train_pids
    # number of object classes
    if hasattr(dataset, "num_train_cls"):
        num_superclasses = dataset.num_train_cls
    else:
        num_superclasses = None
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = {}
    if len(dataset.train) > 0:
        train_set = ImageDataset(dataset.train, transforms['train'], ds_type, cfg.INPUT.PIXEL_MEAN, cfg.DATASETS.CROP)
        class_map = None
        if hasattr(dataset, "class_map"):
            class_map = dataset.class_map
        collate_fn = train_collate_fn(class_map, cfg.INPUT.KEEP_RATIO, cfg.INPUT.PIXEL_MEAN)
        if cfg.DATALOADER.PK_SAMPLER == 'on':
            # sample per class and ID
            if cfg.DATALOADER.CLASS_SAMPLER == 'on':
                sampler = RandomClassSampler(dataset.train, ds_type, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_CLASS)
            # sample balanced from all classes
            elif cfg.DATALOADER.CLASS_BALANCED_SAMPLER == 'on':
                sampler = ClassBalancedSampler(dataset.train, ds_type, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            # sample per ID
            else:
                sampler = RandomIdentitySampler(dataset.train, ds_type, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            data_loader['train'] = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=sampler, num_workers=num_workers, collate_fn=collate_fn
            )
        else:
            data_loader['train'] = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=collate_fn
            )

    collate_fn = val_collate_fn(cfg.INPUT.KEEP_RATIO, cfg.INPUT.PIXEL_MEAN)
    if cfg.TEST.PARTIAL_REID == 'off':
        eval_set = ImageDataset(dataset.query + dataset.gallery, transforms['eval'], ds_type, cfg.INPUT.PIXEL_MEAN, cfg.DATASETS.CROP)
        data_loader['eval'] = DataLoader(
            eval_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=collate_fn
        )
    else:
        dataset_reid = init_dataset('partial_reid', root=cfg.DATASETS.ROOT_DIR)
        dataset_ilids = init_dataset('partial_ilids', root=cfg.DATASETS.ROOT_DIR)
        eval_set_reid = ImageDataset(dataset_reid.query + dataset_reid.gallery, transforms['eval'], ds_type, cfg.INPUT.PIXEL_MEAN, cfg.DATASETS.CROP)
        data_loader['eval_reid'] = DataLoader(
            eval_set_reid, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=collate_fn
        )
        eval_set_ilids = ImageDataset(dataset_ilids.query + dataset_ilids.gallery, transforms['eval'], ds_type, cfg.INPUT.PIXEL_MEAN, cfg.DATASETS.CROP)
        data_loader['eval_ilids'] = DataLoader(
            eval_set_ilids, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=collate_fn
        )

    collate_fn = inference_collate_fn(cfg.INPUT.KEEP_RATIO, cfg.INPUT.PIXEL_MEAN)
    num_samples = None
    if cfg.INFERENCE.DO_INFERENCE == 'on':
        if cfg.INFERENCE.SAMPLES > len(dataset.query):
            num_samples = len(dataset.query)
            inference_set = ImageDataset(dataset.query + dataset.gallery, transforms['eval'], ds_type, cfg.INPUT.PIXEL_MEAN, cfg.DATASETS.CROP)
        else:
            num_samples = cfg.INFERENCE.SAMPLES
            queries = [dataset.query[i] for i in np.random.choice(len(dataset.query), num_samples, replace=False)]
            inference_set = ImageDataset(queries + dataset.gallery, transforms['eval'], ds_type, cfg.INPUT.PIXEL_MEAN, cfg.DATASETS.CROP)
        data_loader['inference'] = DataLoader(
            inference_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=collate_fn
        )
    else:
        data_loader['inference'] = None

    return data_loader, len(dataset.query), num_classes, num_samples, num_superclasses
