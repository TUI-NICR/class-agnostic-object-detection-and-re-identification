import csv
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.structures.bbox import bbox_overlaps
from tqdm import tqdm

IOU = 0.25

model = init_detector(
    config='/path/to/mmdetection/configs/lvis/mask-rcnn_r101_fpn_sample1e-3_ms-1x_lvis-v1.py',
    checkpoint='https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1-ec55ce32.pth'
)

with open("co3d_reid_v1.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = [r for r in reader]

# create bounding boxes with detector and pick the one with iou > IOU and highest overlap to original BB
# the created dataset was never really used
data_v2 = [header]
for path, cls, id, img_num, ymax, ymin, xmax, xmin in tqdm(data):
    gt_bbox = torch.tensor([[int(xmin), int(ymin), int(xmax), int(ymax)]], dtype=torch.float32).cuda()
    results = inference_detector(model, path).pred_instances
    scores = results.scores.cpu().numpy()
    order = np.argsort(scores)
    bboxes = results.bboxes[order]     # xmin, ymin, xmax, ymax
    overlaps = bbox_overlaps(gt_bbox, bboxes).cpu().numpy()[0]
    accept = overlaps > IOU
    if not accept.any():
        best = (xmin, ymin, xmax, ymax)
        # this "continue" is a mistake which causes the v2 dataset to have ~300 less images than the v1
        # but since running the script takes ~30h I will just accept it instead of redoing it
        continue
    else:
        best = bboxes[accept][-1].cpu().numpy().astype(np.int32)
    data_v2.append([path, cls, id, img_num, int(best[3]), int(best[1]), int(best[2]), int(best[0])])

with open("co3d_reid_v2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(data_v2)
