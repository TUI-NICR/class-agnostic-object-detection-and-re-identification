# encoding: utf-8


import os.path as osp
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path, mode="RGB"):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert(mode)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image ReID Dataset"""

    def __init__(self, dataset, transform=None, ds_type="object", mean=None, crop="on"):
        assert ds_type in ["person", "object", "object-masked"]
        if ds_type == "object-masked":
            assert mean is not None
        self.ds_type = ds_type
        self.dataset = dataset
        self.transform = transform
        if mean:
            self.mean = tuple(int(v*255) for v in mean)
        self.crop = True if crop == "on" else False
        if self.crop:
            assert ds_type not in ["person"]
        else:
            assert ds_type not in ["object-masked"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        other = []
        transform_kwargs = {}
        if self.ds_type == "object":
            img_path, pid, camid, obj_class, img_num, ymax, ymin, xmax, xmin = self.dataset[index]
            img = read_image(img_path)
            if self.crop:
                img = img.crop((xmin, ymin, xmax, ymax))
            other = [obj_class]
            transform_kwargs["bbox"] = (xmin, ymin, xmax, ymax)
        elif self.ds_type == "object-masked":
            img_path, img_mask_path, pid, camid, obj_class, img_num, ymax, ymin, xmax, xmin = self.dataset[index]
            img = read_image(img_path)
            mask = read_image(img_mask_path, "L")
            img = img.crop((xmin, ymin, xmax, ymax))
            mask = mask.crop((xmin, ymin, xmax, ymax))
            # cut out segmentation mask from image
            img = Image.composite(img, Image.new("RGB", img.size, self.mean), mask)
            other = [obj_class]
            transform_kwargs["bbox"] = (xmin, ymin, xmax, ymax)
        elif self.ds_type == "person":
            img_path, pid, camid = self.dataset[index]
            img = read_image(img_path)
        else:
            raise NotImplementedError

        if self.transform is not None:
            img = self.transform(img, **transform_kwargs)

        return img, pid, camid, img_path, other
