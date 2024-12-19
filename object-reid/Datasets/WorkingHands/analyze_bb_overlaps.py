import os
import json
import numpy as np
import torch
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from collections import defaultdict

PATH = "/path/to/Working_Hands"
IOU_THR = 0.0


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
ious = defaultdict(lambda: defaultdict(dict))
ious_all = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for ps in os.listdir(PATH):
    p = os.path.join(PATH, ps)
    for o in os.listdir(p):
        p = os.path.join(PATH, ps, o)
        classes = class_restrictions[ps].get(o)
        if classes is None:
            continue
        with open(os.path.join(p, "class_names.txt"), "r") as f:
            class_ids = {e.strip(): i for i, e in enumerate(f)}
        bbs_cls = defaultdict(list)
        for img_n in sorted(os.listdir(os.path.join(p, "RGBImages"))):
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
                bbs_cls[cls].append((img_n, (xmin, ymin, xmax, ymax)))
        for cls_, bbs in bbs_cls.items():
            if len(bbs) > 0:
                bbs = list(zip(*bbs))
                bbs_ = torch.tensor(bbs[1])
                ious_ = box_iou(bbs_, bbs_).numpy()
                for i, iou in zip(bbs[0][1:], ious_.diagonal(offset=1)):
                    ious[cls_][ps + "__" + o][i] = {"iou" : float(iou)}
                for i, iou_ in zip(bbs[0], ious_):
                    for j, iou in zip(bbs[0], iou_):
                        ious_all[cls_][ps + "__" + o][i][j] = {"iou" : float(iou)}

with open("TMP_WH_ious.json", "w") as f:
    json.dump(ious_all, f)


df = pd.DataFrame.from_dict({
    (i, j, k): ious[i][j][k]
        for i in ious.keys()
            for j in ious[i].keys()
                for k in ious[i][j].keys()
    },
    orient="index"
)
df = df.rename_axis(["class", "vid", "num"])
ax = df.hist(by="class", bins=20, figsize=(15, 20))
plt.savefig("wh_consec_ious.pdf")


with open("TMP_WH_ious.json", "r") as f:
    ious_all = json.load(f)

df = pd.DataFrame.from_dict({
    (i, j, k, m): ious_all[i][j][k][m]
        for i in ious_all.keys()
            for j in ious_all[i].keys()
                for k in ious_all[i][j].keys()
                    for m in ious_all[i][j][k].keys()
    },
    orient="index"
)
df = df.rename_axis(["class", "vid", "num_1", "num_2"])

results = defaultdict(dict)
for cls_, df_cls in df.groupby("class"):
    for vid, df_vid in df_cls.groupby("vid"):
        data = pd.pivot_table(
            data=df_vid.reset_index(),
            values="iou",
            index="num_1",
            columns="num_2"
        )
        inds = data.index.values
        values = data.values
        bad_values = values > IOU_THR
        bad_values[np.arange(bad_values.shape[0]), np.arange(bad_values.shape[1])] = False
        while (bad_values.any()):
            sums = bad_values.sum(axis=0)
            i = sums.argmax()
            inds = np.concatenate((inds[:i], inds[i+1:]))
            tmp = np.concatenate((bad_values[:i], bad_values[i+1:]))
            bad_values = np.concatenate((tmp[:, :i], tmp[:, i+1:]), axis=1)
        results[cls_][vid] = (len(inds), list(inds))

with open("non_overlap_imgs.json", "w") as f:
    json.dump(results, f)
