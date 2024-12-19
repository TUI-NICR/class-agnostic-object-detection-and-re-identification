import numpy as np
import pandas as pd
import os
import cv2
import csv
import json
from tqdm import tqdm
from collections import defaultdict

ROOT = "/path/to/CO3D"
SEED = 1337
IMG_PER_ID = 20
IOU_THRESHOLD = 0.0
SEQUENCE_THRS = 20


with open("/path/to/object-reid/Datasets/CO3D/co3d_reid_v1.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    old_co3d = [r for r in reader]
old_co3d = {e[0]: e[4:] for e in old_co3d}

with open("TMP_co3d_consec_ious_2.json", "r") as f:
    ious = json.load(f)

df_iou = pd.DataFrame.from_dict({
    (i, j, int(k)): ious[i][j][k]
        for i in ious.keys()
            for j in ious[i].keys()
                for k in ious[i][j].keys()
    },
    orient="index"
)
df_iou = df_iou.rename_axis(["class", "id", "num"])

ious = defaultdict(dict)
df_iou["iou"] = np.where(df_iou["iou"] > IOU_THRESHOLD, True, False)
for class_, df_class in df_iou.groupby("class"):
    for id_, df_id in df_class.groupby("id"):
        df_id = df_id.reset_index()
        img_nums = df_id["num"]
        inds = np.squeeze(np.argwhere(df_id["iou"] == False), axis=1)
        if len(inds) > 0:
            lengths = np.concatenate((inds[:1]+1, inds[1:] - inds[:-1], len(df_id["iou"]) - inds[-1:]))
            ind_nums = img_nums[inds].values
        else:
            lengths = np.array([len(df_id["iou"])])
            ind_nums = np.array([], dtype=np.int64)
        num_to_len = {
            i for len_, num_0, num_1 in zip(lengths, np.concatenate(([0], ind_nums)), np.concatenate((ind_nums, [img_nums.values[-1]+1])))
                for i in range(num_0, num_1) if len_ >= SEQUENCE_THRS
        }
        ious[class_][id_] = num_to_len


rng = np.random.default_rng(SEED)
c_id = 0
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
        # filter by sequence length
        image_nums = [int(img.split(".")[0].split("frame")[-1]) for img in images]
        images = [img for num, img in zip(image_nums, images) if num in ious[class_name].get(id_, set())]
        # filter faulty masks
        images_and_masks = []
        for img_path in images:
            img_path_full = os.path.join(d_imgs, img_path)
            if img_path_full in old_co3d:
                images_and_masks.append((img_path_full, None))
            else:
                img_name = img_path.split(".")[0]
                mask = None
                for ending in [".jpg", ".png"]:
                    mask_path = os.path.join(d_id, "masks", img_name + ending)
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path)
                if mask is not None:
                    mask_inds = np.argwhere(mask[..., 0] > 0)
                    if mask_inds.size != 0:
                        images_and_masks.append((img_path_full, mask_inds))
        # sample IMG_PER_ID images per ID
        if IMG_PER_ID != "ALL":
            if IMG_PER_ID > len(images_and_masks):
                image_choices = images_and_masks
            else:
                inds = rng.choice(len(images_and_masks), IMG_PER_ID, replace=False)
                image_choices = [images_and_masks[i] for i in inds]
        else:
            image_choices = images_and_masks
        for i, (img_path_full, mask_inds) in enumerate(image_choices):
            if img_path_full in old_co3d:
                ymax, ymin, xmax, xmin = old_co3d[img_path_full]
                ymax, ymin, xmax, xmin = int(ymax), int(ymin), int(xmax), int(xmin)
            else:
                xmax = np.max(mask_inds[:, 1], axis=0)
                xmin = np.min(mask_inds[:, 1], axis=0)
                ymax = np.max(mask_inds[:, 0], axis=0)
                ymin = np.min(mask_inds[:, 0], axis=0)
            data.append([img_path_full, class_name, c_id, i, ymax, ymin, xmax, xmin])
        c_id += 1

    with open("co3d_reid_v10.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class", "id", "img_num", "ymax", "ymin", "xmax", "xmin"])
        writer.writerows(data)
