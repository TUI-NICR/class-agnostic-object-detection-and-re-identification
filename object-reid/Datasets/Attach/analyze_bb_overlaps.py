import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from collections import defaultdict

PATH = "/path/to/Attach/labels"
IOU_THR = 0.0

if False:
    ious_all = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    ious = defaultdict(lambda: defaultdict(dict))
    # collect all images
    for t in os.listdir(PATH):
        if "front" in t:
            view = "front"
        elif "side" in t:
            view = "side"
        elif "spike" in t:
            view = "spike"
        else:
            continue
        p = os.path.join(PATH, t)
        labels = os.listdir(p)
        bbs_cls = defaultdict(list)
        for lab in labels:
            p = os.path.join(PATH, t, lab)
            with open(p, "r") as f:
                data = json.load(f)
            img_p = data["imagePath"]
            for s in data["shapes"]:
                label = s["label"]
                if label in ["Wrench", "Hammer", "Screwdriver"]:
                    points = np.array(s["points"])
                    xmax = np.max(points[:, 0])
                    xmin = np.min(points[:, 0])
                    ymax = np.max(points[:, 1])
                    ymin = np.min(points[:, 1])
                    bbs_cls[label].append((img_p, (xmin, ymin, xmax, ymax)))
        for cls_, bbs in bbs_cls.items():
            if len(bbs) > 0:
                bbs = list(zip(*bbs))
                bbs_ = torch.tensor(bbs[1])
                ious_ = box_iou(bbs_, bbs_).numpy()
                for i, iou in zip(bbs[0][1:], ious_.diagonal(offset=1)):
                    ious[cls_][t][i] = {"iou" : float(iou)}
                for i, iou_ in zip(bbs[0], ious_):
                    for j, iou in zip(bbs[0], iou_):
                        ious_all[cls_][t][i][j] = {"iou" : float(iou)}

    with open("TMP_Attach_ious.json", "w") as f:
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
    ax = df.hist(by="class", bins=20, figsize=(15, 5))
    plt.savefig("attach_consec_ious.pdf")


with open("TMP_Attach_ious.json", "r") as f:
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
