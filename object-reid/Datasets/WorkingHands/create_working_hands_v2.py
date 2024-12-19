import os
import numpy as np
import csv
import cv2
import json

PATH = "/path/to/Working_Hands"
SEED = 1337
rng = np.random.default_rng(SEED)

MIN_AREA = 1024

# Manually determined classes that can be used in each video without
#  confusing two objects of the same class.
class_restrictions = {
    "sha": {
        "1_1": ["pliers"],
        "1_2": ["glue"],
        "2_1": ["measure"],
        "2_2": ["pencil"],
        "4_1": [("ruler", 0, 28), ("pencil", 0, 28), ("saw", 29, 150)],
        "cutter": ["cutter"],
        "eraser": ["eraser"]
    },
    "xiaoling": {
        "1_1": ["ruler", "cutter"],
        "1_2": [("pencil", 0, 49), ("eraser", 50, 150)],
        "3_2": ["ratchet"],
        "4_1": ["saw"],
        "4_2": ["scissors"]
    },
    "zhi": {
        "1_1": ["cutter"],
        "1_2": ["eraser"],
        "2_1": ["hammer"],
        "2_2": ["hammer", "pliers"],
        "3_1": ["measure"],
        "3_2": ["ruler", "pencil"],
        "6_2": ["scissors"],
        "glue": ["glue"]
    }
}

# collect images for each class
images = {}
for ps in os.listdir(PATH):
    p = os.path.join(PATH, ps)
    for o in os.listdir(p):
        p = os.path.join(PATH, ps, o)
        classes = class_restrictions[ps].get(o)
        if classes is None:
            continue
        with open(os.path.join(p, "class_names.txt"), "r") as f:
            class_ids = {e.strip(): i for i, e in enumerate(f)}
        for img_n in os.listdir(os.path.join(p, "RGBImages")):
            seg_p = os.path.join(p, "SegmentationClass", img_n)
            img_p = os.path.join(p, "RGBImages", img_n)
            seg = cv2.imread(seg_p)
            for cls in classes:
                if isinstance(cls, tuple):
                    cls, start, stop = cls
                    nr = int(img_n.split("image_")[-1].split(".png")[0])
                    if nr < start or nr > stop:
                        continue
                cls_id = class_ids[cls]
                mask_inds = np.argwhere(seg[..., 0] == cls_id)
                if mask_inds.size == 0:
                    continue
                xmax = np.max(mask_inds[:, 1], axis=0)
                xmin = np.min(mask_inds[:, 1], axis=0)
                ymax = np.max(mask_inds[:, 0], axis=0)
                ymin = np.min(mask_inds[:, 0], axis=0)
                if cls not in images:
                    images[cls] = []
                images[cls].append([img_p, ymax, ymin, xmax, xmin])

with open("non_overlap_imgs.json", "r") as f:
    data = json.load(f)
allowed_imgs = {cls_: [os.path.join(PATH, *vid.split("__"), "RGBImages", img) for vid, imgs in vids.items() for img in imgs[1]] for cls_, vids in data.items()}

# sample 15 images for each class
header = ["path", "class", "id", "img_num", "ymax", "ymin", "xmax", "xmin"]
data = [header]
for id, (cls, imgs) in enumerate(images.items()):
    imgs = [img for img in imgs if img[0] in allowed_imgs[cls]]
    imgs = [img for img in imgs if (img[1] - img[2]) * (img[3] - img[4]) >= MIN_AREA]
    if len(imgs) < 15:
        continue
    inds = rng.choice(len(imgs), 15, replace=False)
    for i, ind in enumerate(inds):
        e = imgs[ind]
        entry = [e[0], cls, id, i, e[1], e[2], e[3], e[4]]
        data.append(entry)

with open("WorkingHands_v2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(data)
