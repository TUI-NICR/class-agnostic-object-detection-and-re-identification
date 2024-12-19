# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .coco import CocoDataset
from .lvis_ca import LVISCADataset
from .api_wrappers import COCO


@DATASETS.register_module()
class LVISVDataset(LVISCADataset):
    """LVIS v0.5 dataset for detection."""

    METAINFO = {
        'classes':
        # ('object'),
        ('background', 'person'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        # [(255, 255, 0)]
        [(0, 0, 0), (255, 255, 255)]
    }

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        try:
            import lvis
            if getattr(lvis, '__version__', '0') >= '10.5.3':
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',  # noqa: E501
                    UserWarning)
            from lvis import LVIS
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'  # noqa: E501
            )
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.lvis = LVIS(local_path)
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.lvis.cat_img_map)

        img_ids = self.lvis.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.lvis.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            if raw_img_info['file_name'].startswith('COCO'):
                # Convert form the COCO 2014 file naming convention of
                # COCO_[train/val/test]2014_000000000000.jpg to the 2017
                # naming convention of 000000000000.jpg
                # (LVIS v1 will fix this naming issue)
                raw_img_info['file_name'] = raw_img_info['file_name'][-16:]
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.lvis.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.lvis

        return data_list


#LVISVDataset = LVISVDataset
#DATASETS.register_module(name='LVISVDataset', module=LVISVDataset)


@DATASETS.register_module()
class LVISCOCODataset(LVISVDataset):
    """LVIS v1 dataset for detection."""

    METAINFO = {
        'classes':
        # ('object'),
        ('background', 'person'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        # [(255, 255, 0)]
        [(0, 0, 0), (255, 255, 255)]
    }

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        try:
            import lvis
            if getattr(lvis, '__version__', '0') >= '10.5.3':
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',  # noqa: E501
                    UserWarning)
            from lvis import LVIS
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'  # noqa: E501
            )
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.lvis = LVIS(local_path)  # LVIS(local_path)
        self.cat_ids = [1]  # self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.lvis.cat_img_map)

        img_ids = self.lvis.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.lvis.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            # coco_url is used in LVISv1 instead of file_name
            # e.g. http://images.cocodataset.org/train2017/000000391895.jpg
            # train/val split in specified in url
            raw_img_info['file_name'] = raw_img_info['coco_url'].replace(
                'http://images.cocodataset.org/', '')
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.lvis.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            if parsed_data_info is not None:
                data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.lvis

        return data_list
