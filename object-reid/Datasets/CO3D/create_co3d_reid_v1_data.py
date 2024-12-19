import numpy as np
import os
import cv2
import csv
from tqdm import tqdm

ROOT = "/path/to/CO3D"
SEED = 1337
IMG_PER_ID = 20

with open("co3d_reid_v1.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "class", "id", "img_num", "ymax", "ymin", "xmax", "xmin"])

c_load_error = 0
c_mask_error = 0

rng = np.random.default_rng(SEED)
c_id = 0
c_total = 0
data = []
for d in tqdm(os.listdir(ROOT)):
    d_class_super = os.path.join(ROOT, d)
    class_name = os.listdir(d_class_super)[0]
    d_class = os.path.join(d_class_super, class_name)
    for id_ in os.listdir(d_class):
        d_id = os.path.join(d_class, id_)
        if not os.path.isdir(d_id):
            continue
        d_imgs = os.path.join(d_id, "images")
        images = os.listdir(d_imgs)
        # sample IMG_PER_ID images per ID
        if IMG_PER_ID != "ALL":
            if IMG_PER_ID > len(images):
                image_choices = images
            else:
                inds = rng.choice(len(images), IMG_PER_ID, replace=False)
                image_choices = [images[i] for i in inds]
        else:
            image_choices = images
        for i, img_path in enumerate(image_choices):
            img_name = img_path.split(".")[0]
            mask = None
            for ending in [".jpg", ".png"]:
                mask_path = os.path.join(d_id, "masks", img_name + ending)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path)
            if mask is None:
                c_load_error += 1
                continue
            mask_inds = np.argwhere(mask[..., 0] > 0)
            if mask_inds.size == 0:
                c_mask_error += 1
                continue
            xmax = np.max(mask_inds[:, 1], axis=0)
            xmin = np.min(mask_inds[:, 1], axis=0)
            ymax = np.max(mask_inds[:, 0], axis=0)
            ymin = np.min(mask_inds[:, 0], axis=0)
            data.append([os.path.join(d_imgs, img_path), class_name, c_id, i, ymax, ymin, xmax, xmin])
            c_total += 1
        c_id += 1

with open("co3d_reid_v1.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(f"{c_total=}, {c_load_error=}, {c_mask_error=}")
