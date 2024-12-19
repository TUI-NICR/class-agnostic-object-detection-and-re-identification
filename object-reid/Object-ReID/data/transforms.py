# encoding: utf-8
import inspect
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from numpy.random import default_rng

from survey.data.transforms import RandomErasing


def build_transforms(cfg):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    transforms = {}
    if cfg.TEST.PARTIAL_REID == 'off':
        t = []
        if cfg.INPUT.BOX_CROP == 'on':
            t.append(RandomBoxCrop())
        if cfg.INPUT.KEEP_RATIO == 'on':
            assert cfg.INPUT.CROP != 'on'
            t.append(T.Resize(cfg.INPUT.IMG_SIZE[0], max_size=cfg.INPUT.IMG_SIZE[1]))
        else:
            t.append(T.Resize(cfg.INPUT.IMG_SIZE))
        t += [
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.RandomVerticalFlip(p=cfg.INPUT.V_PROB),
            T.RandomPerspective(p=cfg.INPUT.P_PROB)
        ]
        if cfg.INPUT.ROTATE == 'on':
            t.append(T.RandomRotation(degrees=(-cfg.INPUT.ROTATION, cfg.INPUT.ROTATION)))
        if cfg.INPUT.CROP == 'on':
            t += [
                T.Pad(cfg.INPUT.PADDING, [int(v*255) for v in cfg.INPUT.PIXEL_MEAN]),
                T.RandomCrop(cfg.INPUT.IMG_SIZE)
            ]
        t += [
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ]
        transforms['train'] = Compose(t)
    else:
        transforms['train'] = Compose([
            T.Resize(cfg.INPUT.IMG_SIZE),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.RandomResizedCrop(size=256, scale=(0.5, 1.0), ratio=(1.0, 3.0)),
            T.Resize(cfg.INPUT.IMG_SIZE),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    t = []
    if cfg.INPUT.KEEP_RATIO == 'on':
        t.append(T.Resize(cfg.INPUT.IMG_SIZE[0], max_size=cfg.INPUT.IMG_SIZE[1]))
    else:
        t.append(T.Resize(cfg.INPUT.IMG_SIZE))
    t += [
        T.ToTensor(),
        normalize_transform
    ]
    transforms['eval'] = Compose(t)

    return transforms


class Compose(T.Compose):
    """
    Modified torchvision Compose to allow kwargs in call.
    """
    def __call__(self, img, **kwargs):
        for t in self.transforms:
            sig = inspect.signature(t).parameters
            kwargs_ = {k: kwargs[k] for k in kwargs if k in sig}
            img = t(img, **kwargs_)
        return img


class RandomBoxCrop(object):
    """
    Calculates the set of quadratic boxes of maximum size that still fit the image
    while also containing the objects bounding box. Chooses one of these boxes at
    random and crops the image with it.
    """
    def __init__(self, random_seed: int =None) -> None:
        self.rng = default_rng(seed=random_seed)

    def __call__(self, img, bbox):
        width, height = img.size
        xmin, ymin, xmax, ymax = bbox
        side = max(width, height)
        # bounds for top left corner of box
        left_bound = max(0, xmax - side)
        right_bound = min(width - side, xmin)
        top_bound = max(0, ymax - side)
        bottom_bound = min(height - side, ymin)
        x = int(self.rng.random() * (right_bound - left_bound) + left_bound)
        y = int(self.rng.random() * (bottom_bound - top_bound) + top_bound)
        img = img.crop((x, y, x + side, y + side))
        return img


class BatchPad(object):
    def __init__(self, target_size, fill=0):
        self.target_size = target_size
        if isinstance(fill, (float, int)):
            self.fill = [fill]*3
        elif isinstance(fill, (list, tuple)):
            self.fill = fill
        else:
            raise TypeError(f"fill needs to be number or tuple, got {type(fill)}!")

    def __call__(self, img):
        h, w = img.shape[-2:]
        h_padding = (self.target_size[0] - w) / 2
        v_padding = (self.target_size[1] - h) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
        return torch.stack([F.pad(img[..., i, :, :], padding, self.fill[i], "constant") for i in range(3)], img.dim() - 3)
