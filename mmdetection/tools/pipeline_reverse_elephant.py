import argparse
import cv2
import torch
import os
import numpy as np
import time

from pycocotools.cocoeval import COCOeval as EVAL
from pycocotools.coco import COCO

from ReID.tools.caod_interface_script import reidentify
from ReID.config import cfg
from ReID.modeling import build_model
from mmdet_tools.det_reverse_ele import detection_reverse as detection_reverse_ele

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model as build_det_model


def parse_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        '--config-reid',
        type=str,
        default='/path/to/models/reidentification/ReID/tools/caod_interface_cfg.yml',
        help='path to config of reid model',
    )
    parser.add_argument(
        '--config-det',
        type=str,
        default='/path/to/models/Elephant-of-object-detection/training_configs/faster_rcnn_R_50_FPN.yaml',
        help='path to config of detection model',
    )
    parser.add_argument(
        '--img-filepath',
        type=str,
        default='/path/to/attach_benchmark/imgs_full/table/',
        help='filepath to input images'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.10,
        help='only draw bboxes with a confidence score higher than this threshold',
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.45,
        help='delete all boxes with iou > threshold',
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='/path/to/ReID/reverse/',
        help='filepath where preds should be stored'
    )
    parser.add_argument(
        '--csv-out',
        type=str,
        default='/path/to/ReID/csv_results/',
        help='filepath where preds should be stored'
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=640,
        help="crop height",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=640,
        help="crop width",
    )
    parser.add_argument(
        "--comparison-images",
        type=str,
        default='/path/to/attach_benchmark/reid/comparison_all/',#'/path/to/ReID/comparison_images/table_hammer/',
        help="path from where the comparison images for the reidentification task should get loaded from",
    )
    '/path/to/attach_benchmark/reid/table.json'
    parser.add_argument(
        '--eval-annotation',
        type=str,
        default='/path/to/attach_benchmark/reid/table_all.json',
        help='path to anns for eval',
    )
    parser.add_argument(
        "--eval",
        action='store_true',
        default=True,
        help="if evaluation should be done, needs annotations",
    )
    return parser.parse_args()


def draw_bboxes(image, pred_list, threshold, colors):

    if len(pred_list) > 0:
        # determine the number of predictions
        n = len(pred_list)

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        cmap = []
        # create different colours for all bboxes
        for i in range(n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3
            cmap.append((r, g, b))
    img_out = image
    # iterate through all predictions
    for i, pred in enumerate(pred_list):
        bbox = pred['bbox']
        slice_y = slice(bbox[1], bbox[3])
        slice_x = slice(bbox[0], bbox[2])
        cut = image[slice_y, slice_x, ...]
        if cut.shape[0] != 0 and cut.shape[1] != 0:
            score = pred['score']
            # exclude bboxes with score lower than a predefined threshold
            if score > threshold: # threshold
                color = cmap[i]
                # draw the bbox into the image
                img_out = cv2.rectangle(
                    img_out,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color = color, thickness=4
                )
                cv2.putText(
                    img_out,
                    f'{i+1}',
                    (int(bbox[0]), int(bbox[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=cmap[i],
                    thickness=3
                )
    return img_out


def main(args):

    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'

    # path to the input image or a directory with images 
    input_path = args.img_filepath

    # initialize reidentification model
    config_path_reid = args.config_reid

    cfg.merge_from_file(config_path_reid)
    cfg.freeze()
    # create model structure
    # num_classes does not matter here because classifier is only used during training
    reid_model = build_model(cfg, num_classes=0)
    # cuda by default
    device = cfg.MODEL.DEVICE
    if 'cpu' not in device:
        reid_model.to(device=device)
    # load weights
    reid_model.load_param(cfg.TEST.WEIGHT)
    reid_model.eval()

    # init det model
    config_path_det = args.config_det
    cfg_det = get_cfg()
    cfg_det.merge_from_file(config_path_det)
    cfg_det.freeze()
    det_model = build_det_model(cfg_det)
    det_model.to(device)
    DetectionCheckpointer(det_model).load(cfg_det.MODEL.WEIGHTS)
    det_model.eval()

    # collect all iamges that are going to be processed in a list
    if not os.path.isdir(input_path):
        targets = [input_path]
    else:
        targets = [
            f for f in os.listdir(input_path) if not os.path.isdir(os.path.join(input_path, f))
        ]
        targets = [os.path.join(input_path, f) for f in targets]

    # load comparison images
    comparison_path = args.comparison_images
    if not os.path.isdir(comparison_path):
        comparison_images = [comparison_path]
    else:
        comparison_images = [
            f for f in os.listdir(comparison_path) if not os.path.isdir(os.path.join(comparison_path, f))
        ]
        comparison_images = [os.path.join(comparison_path, f) for f in comparison_images]
    
    reid_targets = []
    for c in comparison_images:
        img = cv2.imread(c)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        reid_targets.append(img)

    # predictions array for later evaluation
    predictions = []
    p_all = []

    # process the images
    for iter, image_path in enumerate(targets):
        # create 'window boxes' for the reidentification module
        boxes = []
        boxes_start = []
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        #test
        x1, y1, crop_w, crop_h = [550, 550, 1600, 1020]
        slice_y = slice(y1, y1+crop_h)
        slice_x = slice(x1, x1+crop_w)
        img_ori = image.copy()
        image = image[slice_y, slice_x, ...]
        height, width, _ = image.shape
        mask_params = [x1, y1]
        # testend
        y_start = 0
        x_start = 0
        crop_height = args.crop_height
        # sliding window over image, check for image bounds
        # TODO: change second condition?
        it = 0
        while (y_start < height and crop_height >= (args.crop_height/2)):
            x_start = 0
            crop_width = args.crop_width
            while (x_start < width and crop_width >= (args.crop_width/2)):
                # determine next crop
                slice_y = slice(y_start, y_start+crop_height)
                slice_x = slice(x_start, x_start+crop_width)
                crop_image = image[slice_y, slice_x, ...]
                boxes.append(crop_image)
                boxes_start.append((x_start, y_start, crop_width, crop_height))
                it += 1

                # next start coordinates for sliding window
                x_start += int(crop_width/2)
                # edge treatment, if next crop doesnt fit in image
                # crop_width is set to default again in next loop pass
                if x_start+crop_width > width:
                    crop_width = width - x_start

            y_start += int(crop_height/2)
            # edge treatment, if next crop doesnt fit in image
            if y_start+crop_height > height:
                crop_height = height - y_start

        # reidentification module
        distmat = reidentify(reid_model, boxes, reid_targets, args, device, cfg=cfg)
        # do postprocess
        mins_ = []
        n = 4
        for i in range(len(reid_targets)):
            min = np.argsort(distmat[i])[:n]
            # or use arpartition, is faster, but first n elements may not be ordered
            #min = np.argpartition(distmat[i], 5)[:n]
            mins_.append(min)
        # detection module
        start1 = time.time()
        boxes, preds = detection_reverse_ele(args, ort_sess=None, image_path=image_path, cropped_boxes=boxes, boxes_start=boxes_start, mins=mins_, det_model=det_model, mask_params=mask_params)
        end1 = time.time()
        inference_time = end1 - start1
        print(f"Inference time det: {inference_time} seconds")
        # reidentification module
        # load comparison images
        distmat = reidentify(reid_model, boxes, reid_targets, args, device, cfg=cfg)


        mins_ = []
        n = 10
        colors = [(255, 0, 0), (255, 51, 51), (255, 102, 102), (255, 153, 153), (255, 204, 204),\
                (255, 128, 128), (255, 77, 77), (255, 26, 26), (255, 179, 179), (255, 230, 230)]
        for i in range(len(reid_targets)):
            min = np.argsort(distmat[i])[:n]
            # or use arpartition, is faster, but first n elements may not be ordered
            #min = np.argpartition(distmat[i], 5)[:n]
            mins_.append(min)

        img_pt = image_path
        _, ext = os.path.splitext(image_path)
        img_id = image_path.split('/')[-1].replace(ext, '')
        output_path_reid = f'{args.output_path}/reid_boxes/{img_id}'
        os.makedirs(output_path_reid, exist_ok=True)
        
        for i, min in enumerate(mins_):
            img_ = cv2.imread(img_pt)
            pred_list = []
            for j, ele in enumerate(min):
                pred = {}
                pred['image_id'] = int(img_id)
                pred['category_id'] = 1
                pred['score'] = preds[ele]['score']
                b = preds[ele]['bbox'].copy()
                # change bbox values to height, with
                b[2] = b[2]-b[0]
                b[3] = b[3]-b[1]
                pred['bbox'] = b.tolist()
                predictions.append(pred)
                break
                bbox = preds[ele]
                pred_list.append(bbox)
                #cv2.imwrite(f'{output_path_reid}/test_{i}{j}.png', boxes[ele])
            #img_out = draw_bboxes(img_, pred_list, 0.0, colors)
            #cv2.imwrite(f'{output_path_reid}/_{i}.png', img_out)

        # just for testing
        for p in preds:
            p['image_id'] = int(img_id)
            p['category_id'] = 1
            b = p['bbox'].copy()
            # change bbox values to height, with
            b[2] = b[2]-b[0]
            b[3] = b[3]-b[1]
            p['bbox'] = b.tolist()
        p_all += preds

    # evaluation of boxes after detection
    if args.eval:
        gt_path = args.eval_annotation
        coco_gt = COCO(gt_path)

        coco_dt = coco_gt.loadRes(p_all)
        coco_eval = EVAL(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # evaluation of boxes after ReID
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = EVAL(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == "__main__":
    args = parse_args()
    main(args)
