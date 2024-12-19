# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
import torch

from copy import deepcopy
from typing import List, Optional, Union

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def sliding_window(dataset_name, size):
    ds = DatasetCatalog.get(dataset_name)
    classes = MetadataCatalog.get(dataset_name).thing_classes
    colors = MetadataCatalog.get(dataset_name).thing_colors
    step = size // 2
    dataset = []
    for img in ds:
        for i, x in enumerate(range(0, img["width"], step)):
            for j, y in enumerate(range(0, img["height"], step)):
                if img["width"] - x < step or img["height"] - y < step:
                    continue
                entry = deepcopy(img)
                entry["x_point"] = x
                entry["y_point"] = y
                entry["w_size"] = size
                entry["image_id"] = img["image_id"] * 10_000 + i * (img["width"] // step) + j
                for ann in entry["annotations"]:
                    bbox = BoxMode.convert(ann["bbox"], ann["bbox_mode"], BoxMode.XYWH_ABS)
                    bbox[0] -= x
                    bbox[1] -= y
                    ann["bbox_mode"] = BoxMode.XYWH_ABS
                dataset.append(entry)
    name = dataset_name + "_sw"
    DatasetCatalog.register(name, lambda: dataset)
    MetadataCatalog.get(name).thing_classes = classes
    MetadataCatalog.get(name).thing_colors = colors
    return name


class SlidingWindowMapper:

    @configurable
    def __init__(
        self,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str
    ):
        """
        Args:
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg):
        augs = utils.build_augmentation(cfg, is_train=False)
        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        x, y = dataset_dict["x_point"], dataset_dict["y_point"]
        size = dataset_dict["w_size"]
        image = image[y:y+size, x:x+size]
        dataset_dict["height"], dataset_dict["width"] = image.shape[:2]

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)

        image = aug_input.image
        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
