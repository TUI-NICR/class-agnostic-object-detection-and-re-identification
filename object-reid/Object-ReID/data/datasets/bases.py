# encoding: utf-8

from survey.data.datasets.bases import BaseDataset


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset.

    Args:
    - ds_type (str): Type of dataset from ["person", "object", "object-masked"]
    """

    def __init__(self, ds_type="person") -> None:
        super().__init__()
        self.ds_type = ds_type

    def get_imagedata_info(self, data):
        pids, cams, classes = [], [], []
        for e in data:
            if self.ds_type == "object-masked":
                _, _, pid, camid, *other = e
            elif self.ds_type in ["person", "object"]:
                _, pid, camid, *other = e
            else:
                raise NotImplementedError
            if len(other) > 0:
                classes += [other[0]]
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        classes = set(classes)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_classes = len(classes)
        if num_classes > 0:
            return num_pids, num_imgs, num_cams, num_classes
        else:
            return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, *_ = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, *_ = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, *_ = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")
