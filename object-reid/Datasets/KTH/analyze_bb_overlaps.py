import os
import numpy as np
import xml.etree.ElementTree as ET
import torch
import json
import pandas as pd
from torchvision.ops import box_iou
from collections import defaultdict

PATH = "/path/to/KTH"
IOU_THR = 0.0


images = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for bg in os.listdir(PATH):
    p = os.path.join(PATH, bg, "rgb")
    for lt in os.listdir(p):
        p = os.path.join(PATH, bg, "rgb", lt)
        for obj in os.listdir(p):
            p = os.path.join(PATH, bg, "rgb", lt, obj)
            content = os.listdir(p)
            if len(content) == 2:
                for cam in content:
                    p = os.path.join(PATH, bg, "rgb", lt, obj, cam)
                    imgs = os.listdir(p)
                    imgs = [img for img in imgs if "(1).jpg" not in img]
                    images[bg][lt][obj][cam] = imgs
            else:
                imgs = os.listdir(p)
                imgs = [img for img in imgs if "(1).jpg" not in img]
                images[bg][lt][obj]["NONE"] = imgs

ious_all = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for bg, lts in images.items():
    for lt, objs in lts.items():
        for obj, cams in objs.items():
            for cam, imgs in cams.items():
                bbs = []
                for img in imgs:
                    p = os.path.join(PATH, bg, "rgb", lt, obj, cam, img).replace("/NONE", "")
                    pb = p.replace("rgb", "bboxes").replace(".jpg", ".xml")
                    if not os.path.exists(pb):
                        pb = pb.replace("/Kinect/", "/kinect/")
                    tree = ET.parse(pb)
                    root = tree.getroot()
                    box = root.find("object").find("bndbox")
                    ymax = int(box.find("ymax").text)
                    ymin = int(box.find("ymin").text)
                    xmax = int(box.find("xmax").text)
                    xmin = int(box.find("xmin").text)
                    bbs.append((p, (xmin, ymin, xmax, ymax)))
                if len(bbs) > 0:
                    bbs = list(zip(*bbs))
                    bbs_ = torch.tensor(bbs[1])
                    ious_ = box_iou(bbs_, bbs_).numpy()
                    for i, iou_ in zip(bbs[0], ious_):
                        for j, iou in zip(bbs[0], iou_):
                            ious_all[obj]["__".join([bg, lt, cam])][i][j] = {"iou" : float(iou)}

with open("TMP_KTH_ious.json", "w") as f:
        json.dump(ious_all, f)


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
        results[cls_][vid] = (len(inds), inds)

print(results)
