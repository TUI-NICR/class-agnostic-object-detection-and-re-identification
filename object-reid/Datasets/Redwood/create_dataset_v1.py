import os
import json
import csv
import torch
import numpy as np
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.structures.bbox import bbox_overlaps

import cv2

PATH = "/path/to/Redwood"
SEED = 1337
IMG_PER_ID = 20

rng = np.random.default_rng(SEED)

model = init_detector(
    config='/path/to/mmdetection/configs/lvis/mask-rcnn_r101_fpn_sample1e-3_ms-1x_lvis-v1.py',
    checkpoint='https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1-ec55ce32.pth'
)

with open("categories.json", "r") as f:
    categories = json.load(f)
inverse_categories = {nr: cat for cat, e in categories.items() for nr in e}

invalids = 0
data = [["path", "class", "id", "img_num", "ymax", "ymin", "xmax", "xmin"]]
for id, object_nr in enumerate(tqdm(os.listdir(PATH))):
    cls = inverse_categories.get(object_nr, "UNCLASSED")
    p = os.path.join(PATH, object_nr, "rgb")
    img_ps = os.listdir(p)
    # Pick IMG_PER_ID images randomly for each object
    inds = rng.choice(len(img_ps), IMG_PER_ID, replace=False)
    image_choices = [img_ps[i] for i in inds]
    img_num = 0
    for img_p in image_choices:
        path = os.path.join(p, img_p)
        img = cv2.imread(path)
        # Attempt to create BB for image
        results = inference_detector(model, img).pred_instances
        scores = results.scores.cpu().numpy()
        order = np.argsort(scores)
        bboxes = results.bboxes[order]     # xmin, ymin, xmax, ymax
        height, width = img.shape[:2]
        # criterium 1: box must intersect inner third of image
        x_min = width // 3
        x_max = width // 3 * 2
        y_min = height // 3
        y_max = height // 3 * 2
        center_box = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32).cuda()
        overlaps = bbox_overlaps(center_box, bboxes)[0]
        accept = overlaps > 0
        # criterium 2: box must be at least 1/50 of image size
        img_size = torch.tensor(height * width, dtype=torch.float32).cuda()
        accept = accept & ((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) > img_size / 50)
        if not accept.any():
            invalids += 1
            continue
        best = bboxes[accept][-1].cpu().numpy().astype(np.int32)
        data.append([path, cls, id, img_num, int(best[3]), int(best[1]), int(best[2]), int(best[0])])
        img_num += 1

with open("redwood_reid_v1.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(invalids)  # 189
