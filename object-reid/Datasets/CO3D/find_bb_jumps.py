import os
import torch
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from tqdm import tqdm
from collections import defaultdict

CSV_PATH = "/path/to/toDataset/co3d_reid_v1"
CO3D_PATH = "/path/to/CO3D"

if False:
    ious = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for split in ["train", "test", "query"]:
        df = pd.read_csv(
            os.path.join(CSV_PATH, split + ".csv"), sep=",",
            dtype={
                "path": str,
                "class": str,
                "id": int,
                "img_num": int,
                "xmax": int,
                "xmin": int,
                "ymax": int,
                "ymin": int
            }
        )
        df_by_cls = df.groupby("class")
        for class_, data in tqdm(df_by_cls):
            data_by_id = data.groupby("id")
            for id_, data_ in data_by_id:
                data_.sort_values("img_num", inplace=True)
                bbs = torch.tensor(data_[["xmin", "ymin", "xmax", "ymax"]].to_numpy())
                ious_ = box_iou(bbs, bbs).numpy().diagonal(offset=1)
                for i, iou in enumerate(ious_):
                    ious[split][class_][id_][i] = {"iou" : iou}

    df = pd.DataFrame.from_dict({
        (i, j, k, m): ious[i][j][k][m]
            for i in ious.keys()
                for j in ious[i].keys()
                    for k in ious[i][j].keys()
                        for m in ious[i][j][k].keys()
        },
        orient="index"
    )
    df = df.rename_axis(["split", "class", "id", "num"])
    df = df.loc["train"]
    ax = df.hist(by="class", bins=50, figsize=(30, 30))
    plt.savefig("co3d_v1_ds_consec_ious.pdf")


ious = defaultdict(lambda: defaultdict(dict))
for d in tqdm(os.listdir(CO3D_PATH)):
    d_class_super = os.path.join(CO3D_PATH, d)
    class_name = os.listdir(d_class_super)[0]
    d_class = os.path.join(d_class_super, class_name)
    for id_ in os.listdir(d_class):
        d_id = os.path.join(d_class, id_)
        if not os.path.isdir(d_id):
            continue
        d_imgs = os.path.join(d_id, "images")
        images = sorted(os.listdir(d_imgs), key=lambda x: int(x.split("frame")[1].split(".")[0]))
        bbs = []
        for img_path in images:
            img_name = img_path.split(".")[0]
            mask = None
            for ending in [".jpg", ".png"]:
                mask_path = os.path.join(d_id, "masks", img_name + ending)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path)
            if mask is None:
                continue
            mask_inds = np.argwhere(mask[..., 0] > 0)
            if mask_inds.size == 0:
                continue
            xmax = np.max(mask_inds[:, 1], axis=0)
            xmin = np.min(mask_inds[:, 1], axis=0)
            ymax = np.max(mask_inds[:, 0], axis=0)
            ymin = np.min(mask_inds[:, 0], axis=0)
            img_num = int(img_name.split("frame")[-1])
            bbs.append((img_num, [xmin, ymin, xmax, ymax]))
        if len(bbs) > 0:
            bbs = list(zip(*bbs))
            bbs_ = torch.tensor(bbs[1])
            ious_ = box_iou(bbs_, bbs_).numpy().diagonal(offset=1)
            for i, iou in zip(bbs[0][1:], ious_):
                ious[class_name][id_][i] = {"iou" : float(iou)}
    with open("TMP_co3d_consec_ious_2.json", "w") as f:
        json.dump(ious, f)

df = pd.DataFrame.from_dict({
    (i, j, k): ious[i][j][k]
        for i in ious.keys()
            for j in ious[i].keys()
                for k in ious[i][j].keys()
    },
    orient="index"
)
df = df.rename_axis(["class", "id", "num"])
ax = df.hist(by="class", bins=50, figsize=(30, 30))
plt.savefig("co3d_consec_ious.pdf")
