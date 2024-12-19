#%%
import pickle
import random
import cv2
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval as EVAL
from pycocotools.coco import COCO
import torch
import json

objects = []
with (open("/path/to/benchmark_rtmdet/hand/models/tiny.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

predictions = []
for img in objects[0]:
    id = img['img_id']
    for p, score, bbox in zip(img['pred_instances']['masks'],img['pred_instances']['scores'], img['pred_instances']['bboxes']):
        mask = maskUtils.decode(p)
        pred = {}
        pred['image_id'] = int(id)
        pred['category_id'] = 1
        pred['score'] = score#random.uniform(0.9,1.0)
        pred['segmentation'] = p
        pred['bbox'] = bbox.tolist()
        pred['bbox'][2] = pred['bbox'][2] -pred['bbox'][0]
        pred['bbox'][3] = pred['bbox'][3] -pred['bbox'][1]
        predictions.append(pred)

gt_path = '/path/to/attach_benchmark/imgs_cropped/hand.json'

coco_gt = COCO(gt_path)
coco_dt = coco_gt.loadRes(predictions)

coco_eval = EVAL(coco_gt, coco_dt, iouType='segm')

#coco_eval.params.catIds = self.cat_ids
#coco_eval.params.imgIds = self.img_ids
#coco_eval.params.maxDets = list(self.proposal_nums)
#coco_eval.params.iouThrs = self.iou_thrs

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
coco_eval = EVAL(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
# %%
