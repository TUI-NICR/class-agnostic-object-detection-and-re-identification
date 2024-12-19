import csv
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.structures.bbox import bbox_overlaps
import torch
import numpy as np
import json


model = init_detector(
    config='/path/to/mmdetection/configs/lvis/mask-rcnn_r101_fpn_sample1e-3_ms-1x_lvis-v1.py',
    checkpoint='https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1-ec55ce32.pth'
)

with open("co3d_reid_v1.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = [r for r in reader]

# analyse IOU of BBs created with object detector to determine an IOU cutoff point for create_co3d_reid_v2_data.py
# the created dataset was never really used and I don't remember the details of this script
bests = []
for path, cls, id, img_num, ymax, ymin, xmax, xmin in tqdm(data):
    gt_bbox = torch.tensor([[int(xmin), int(ymin), int(xmax), int(ymax)]], dtype=torch.float32).cuda()
    results = inference_detector(model, path).pred_instances
    scores = results.scores.cpu().numpy()
    order = np.argsort(scores)
    bboxes = results.bboxes[order]     # xmin, ymin, xmax, ymax
    overlaps = bbox_overlaps(gt_bbox, bboxes).cpu().numpy()[0]
    best_scores = []
    for i in range(0, 100, 5):
        iou = i / 100
        accept = overlaps > iou
        if not accept.any():
            best_scores.append((-1, -1))
            continue
        best_score = scores[order][accept][-1]
        corr_overlap = overlaps[accept][-1]
        best_scores.append((best_score, corr_overlap))
    bests.append(best_scores)

bests = np.array(bests)
invalids = (bests == -1)
invalids_summed = invalids.sum(0)
bests[invalids] = 0
bests_avg = bests.sum(0) / (bests.shape[0] - invalids_summed)

print(bests_avg, invalids_summed)

with open("results_iou.json", "w") as f:
    d = {
        "bests_avg": bests_avg.tolist(),
        "invalids_summed": invalids_summed.tolist()
    }
    json.dump(d, f)
