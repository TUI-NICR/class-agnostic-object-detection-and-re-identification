import argparse
import cv2
import torch
import os
import numpy as np
import onnxruntime as ort
import json
import csv
import matplotlib.pyplot as plt
import time

from ReID.tools import reidentify
from ReID import cfg
from ReID.modeling import build_model

from mmdet.apis import init_detector

from mmdet_tools import detection_reverse
from mmdet_tools import COCOeval as EVAL

from pycocotools.coco import COCO

from itertools import chain


def parse_args():
    parser = argparse.ArgumentParser('')

    # setup 
    parser.add_argument(
        "--reverse",
        action='store_true',
        default=True,
        help="aply reverse pipeline (a first reidentification before detection)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default='/path/to/models/mmdetstuff/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py',
        #or use for RTMDET '/path/to/models/mmdetstuff/mmdetection/configs_rtmdet/rtmdet_tiny_8xb32-300e_attach.py' 
        help="path to config file of the detection model",
    )
    parser.add_argument(
        "--config-reid",
        type=str,
        default='/path/to/models/reidentification/ReID/tools/caod_interface_cfg.yml',
        help="path to config file of the reidentification model",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default='/path/to/setup_pipeline/checkpoints/dino_epoch_20.pth',
        # or use for RTMDET '/path/to/setup_pipeline/checkpoints/rtmdet_epoch_30.pth'
        help="path to checkpoint of the pretrained detection model",
    )
    parser.add_argument(
        "--do-onnx",
        action='store_true',
        default=False,
        help="if the onnx model or the mmdet inference detector should be used",
    )
    parser.add_argument(
        '--onnx-filepath',
        type=str,
        default='/path/to/rtmdet_deploy/test/2/end2end.onnx',
        help='filepath to onnx model',
    )


    # input / output paths
    parser.add_argument(
        '--img-filepath',
        type=str,
        default='/path/to/attach_benchmark/imgs_full/table/',
        help='filepath to input images'
    )
    parser.add_argument(
        "--comparison-images",
        type=str,
        default='/path/to/attach_benchmark/reid/comparison_all/',
        help="path from where the comparison images for the reidentification task should get loaded from",
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='/path/to/ReID/reverse2/',
        help='filepath where preds should be stored'
    )


    # thresholds
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.075,
        help='only draw bboxes with a confidence score higher than this threshold',
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.7,
        help='delete all boxes with iou > threshold',
    )
    parser.add_argument(
        '--iou-threshold-reid1',
        type=float,
        default=0.5,
        help='delete all boxes with iou > threshold',
    )
    parser.add_argument(
        '--threshold-reid1',
        type=float,
        default=1.5,
        help='consider only crops with cos distance <= threshold',
    )
    parser.add_argument(
        '--max-crops-reid1',
        type=int,
        default=3,
        help='maximum number of image crops selected after first reidentification for each comparison object',
    )
    parser.add_argument(
        '--threshold-reid2',
        type=float,
        default=1.0,
        help='consider only bounding boxes with cos distance <= threshold',
    )
    parser.add_argument(
        '--max-boxes-reid2',
        type=int,
        default=5,
        help='number of bounding boxes that get selected after the second reidentification',
    )


    # crop parameter
    parser.add_argument(
        "--crop-height",
        type=int,
        default=650,
        help="crop height",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=650,
        help="crop width",
    )
    parser.add_argument(
        "--crop-image-table",
        action='store_true',
        default=False,
        help="if true only the area where the table is located in the image will be processed",
    )
    parser.add_argument(
        "--downscale-factor",
        type=int,
        default=1,
        help="downscale for first reidentification",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=512,
        help="size of boxes, around the postioins selected after first reidentification",
    )


    # evaluation and visualization
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="if evaluation should be done, needs annotations",
    )
    parser.add_argument(
        '--ground-truth-annotations',
        type=str,
        default='/path/to/attach_benchmark/reid/table_all.json',
        help='filepath to ground truth annotation file'
    )
    parser.add_argument(
        "--visualize",
        action='store_true',
        default=True,
        help="visualize results",
    )

    return parser.parse_args()


def draw_bboxes(image, pred_list, threshold, text, thickness):
    """
    Function that draws Bounding Boxes and can also visualize the crops selected after a possible first reidentification

    :param image: original image
    :param pred_list: predictions to visualize
    :param threshold: threshold that is needed for a prediction to get drawn
    :param text: if true the box gets marked with an indice in the image
    :param thickness: defines how strong the lines of the boxes should be
    :return: image with drawn bounding boxes
    """

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

    # create copy of image
    img_out = image.copy()
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    # iterate through all predictions
    for i, pred in enumerate(pred_list):
        bbox = pred['bbox']
        if thickness==-1:
            # for marking crops after first reidentification
            slice_y = slice(bbox[1], bbox[3])
            slice_x = slice(bbox[0], bbox[2])
        else:
            slice_y = slice(bbox[1], bbox[1] + bbox[3])
            slice_x = slice(bbox[0], bbox[0] + bbox[2])
        cut = image[slice_y, slice_x, ...]
        if cut.shape[0] != 0 and cut.shape[1] != 0:
            score = pred['score']
            # exclude bboxes with score lower than a predefined threshold
            if score > threshold: # threshold
                color = cmap[i]
                if thickness==-1: # case to visualize crops that will get processed in the detection
                    # draw the bbox into the image
                    img_out = cv2.rectangle(
                        img_out, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),color, thickness=thickness
                        )
                    alpha = 0.5
                    # makes the selected crop transparent
                    img_new = cv2.addWeighted(img_out, alpha, image, 1 - alpha, 0)
                else: # case for normal bounding boxes
                    # draw the bbox into the image
                    img_out = cv2.rectangle(img_out, (int(bbox[0]), int(bbox[1])), (int(bbox[2])+int(bbox[0]), int(bbox[3])+int(bbox[1])),
                            color, thickness=thickness
                        )

                    img_new = img_out
                if text:
                    cv2.putText(
                        img_new,
                        f'{i+1}',
                        (int(bbox[0]), int(bbox[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=(0,0,0),#cmap[i],
                        thickness=3
                    )
    # special case for nothing to visualize
    if len(pred_list) == 0:
        img_new = img_out
    return img_new


def eval(gt_path, predictions, csv_path):
    """
    evaluates predictions, computes recall

    :param gt_path: path to ground truth annotations
    :param predictions: the predicted (and selected - reid) bboxes
    :param csv_path: path where results are saved
    :return: pedictions that were matched in the evaluation
    """
    coco_gt = COCO(gt_path)

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = EVAL(coco_gt, coco_dt, iouType='bbox')
    # coco_eval.evaluate() returns the matched ground truth objects and the corresponding matched predictions
    # matched gorund truths is used to compute the recall per class
    # matched detections for visualization (otherwise there are to many detections in the image)
    matched_gts, matched_dts = coco_eval.evaluate()
    coco_eval.accumulate()
    stats = coco_eval.summarize()
    # get recall value for iou=0.5 and 1000 detections
    recall = stats[-2]

    compute_recall_per_class(matched_gts, recall, csv_path)

    return matched_dts


def compute_recall_per_class(matched_gts, recall, csv_path):
    """
    computes recall per class

    :param matched_gts: ground truths that were matched with a prediction
    :param recall: recall value over all clases
    :param csv_path: path where results are saved
    """
    # header and data row for csv out file
    recall_header = ['All']
    recall_data = [recall]
    # remove duplicates from matched gt
    matched_gts = list(set(matched_gts))
    # load gt annotations with classes
    with open('/path/to/attach_benchmark/reid/table_all_with_class.json', 'r') as f:
        data = json.load(f)
    anns = data['annotations']
    # create arrays for metric computation
    count_all = np.zeros(len(data['categories'])) # "tp+fn" counter
    count_tp = np.zeros(len(data['categories']))
    for ann in anns:
        id = ann['id']
        cat = ann['category_id']
        count_all[cat] += 1
        if id in matched_gts:
            # gt was matched in eval, count true positive up for the category
            count_tp[cat] += 1
    # compute recall
    recall_cats = count_tp / count_all
    # write class name and recall to csv
    for i, r in enumerate(recall_cats):
        n = data['categories'][i]['name']
        print(f'{n}: {r}')
        recall_header.append(n)
        recall_data.append(r)
    with open(csv_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(recall_header)
        writer.writerow(recall_data)
    create_chart(recall_header, recall_data, csv_path)


def visualize(matched_dts, out_path, all_img_ids, predictions, images, args, all=False):
    """
    viualizes the predicted boxes of the detection part and the remaining boxes after the reidentification

    :param matched_dts: ids of predictions that were matched with a ground truth
    :param out_path: path were to save the images
    :param all_image_ids: array with image ids - used for image names
    :param predictions: all predictions of detection / after reid
    :param images: copy of original images
    :param args: input args
    :param all: boolean value, if all preds should get painted
    """
    # get matched predictions after second reidentification
    matched_dts = list(set(matched_dts))
    os.makedirs(out_path, exist_ok=True)
    # collect preds for images
    pred_dict_per_image = {}
    for img_id in all_img_ids:
        pred_dict_per_image[str(img_id)] = []
    for i, p in enumerate(predictions):
        if all:
            pred_dict_per_image[str(p['image_id'])].append(p)
        else:
            if i+1 in matched_dts: # get matched dts
                pred_dict_per_image[str(p['image_id'])].append(p)

    for k, v in pred_dict_per_image.items():
        # visualize each image
        # check if predictions are not empty (can be empty if nothing after reid has matched)
        if len(v) > 0:
            # get id
            img_id = v[0]['image_id']
            # get corresponding image
            image = images[str(img_id)]
            vis_det = image.copy()
            path_ = f'{out_path}{k}'
            vis_det = draw_bboxes(vis_det, v, 0.0, False, thickness=4)
            # crop image, so that only the part that was processed is visualized
            if args.crop_image_table:
                x1, y1, crop_w, crop_h = [550, 550, 1600, 1020]
                slice_y = slice(y1, y1+crop_h)
                slice_x = slice(x1, x1+crop_w)
                vis_det= vis_det[slice_y, slice_x, ...]
            # save image
            cv2.imwrite(f'{path_}.png', vis_det)


def create_chart(class_names, recall_values, path):
    """
    creates a bar chart for the recall values per class

    :param class_names: name of classes
    :param recall_values: computed recall values for each class
    :param path: output path
    """
    path, _ = os.path.splitext(path)
    bar_chart = plt.figure(figsize=(6, 4))
    plt.bar(class_names, recall_values, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Recall')
    plt.title('Recall per Class')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')  # Adding grid lines for better readability
    bar_chart.savefig(f'{path}.png', bbox_inches='tight')


def non_maximum_suppression(boxes, iou_threshold):
    """
    Non-Maximum-Suppression for boxes after reid1, code is adapted from MMI course

    :param boxes: list with all predictions
    :param iou_threshold: threshold for intersection over union
    :return: final boxes after non maximum suppression
    """

    final_boxes = []
    suppressed = [False] * len(boxes)
    # sort all boxes after their score
    # boxes = sorted(boxes, key=lambda b: b['score'], reverse=False)

    def area(box):
        """helper function to compute area of a box"""
        a = (box[2]-box[0]) * (box[3]-box[1])
        return a
    
    # iterate through all boxes
    for i, box in enumerate(boxes):
        # check if already suppressed
        if suppressed[i]:
            continue
        
        final_boxes.append(box)

        box_area = area(box)
        for j in range(i + 1, len(boxes)):
            # wurde die Box bereits bei der Verarbeitung einer Box mit
            # hoeherer Konfidenz ausgeschlossen, wird sie uebersprungen
            if suppressed[j]:
                continue
            box_j = boxes[j]
            box_j_area = area(box_j)

            # Ueberlapp (Intersection) zwischen den beiden aktuell betrachteten Boxen berechnen
            # Bedenken Sie dabei auch Faelle, in denen kein Ueberlapp entsteht
            x1 = max(box[0], box_j[0])
            y1 = max(box[1], box_j[1])
            x2 = min(box[2], box_j[2])
            y2 = min(box[3], box_j[3])

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            # Intersection over Union (IoU) berechnen
            intersection = w * h
            union = box_area + box_j_area - intersection
            iou = intersection / union

            # Box j fuer Betrachtungen ausschliessen, falls die IoU den
            # Schwellwert uebersteigt
            if iou > iou_threshold:
                suppressed[j] = True
        
    return final_boxes


def main(args):

    # initialize detection model
    if args.do_onnx:
        onnx_filepath = args.onnx_filepath
        # create the Inference session with the exported ONNX Model
        ort_sess = ort.InferenceSession(onnx_filepath, providers=['CUDAExecutionProvider'])
        det_model = None #dummy
    else:
        config_file =  args.config_file
        checkpoint_file = args.checkpoint_file
        device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        det_model = init_detector(config_file, checkpoint_file, device=device)
        ort_sess = None # dummy

    # path to the input image or a directory with images 
    input_path = args.img_filepath

    # initialize reidentification model
    config_path = args.config_reid

    cfg.merge_from_file(config_path)
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
    config = cfg

    if args.reverse:
        # initialize reid model for first step
        cfg.merge_from_file('/path/to/setup_pipeline/configs/plain_resnet.yml')
        cfg.freeze()
        reid_model1 = build_model(cfg, num_classes=0)
        # cuda by default
        device = cfg.MODEL.DEVICE
        if 'cpu' not in device:
            reid_model1.to(device=device)
        # load weights
        reid_model1.load_param(cfg.TEST.WEIGHT)
        reid_model1.eval()

    # collect all images that are going to be processed in a list
    if not os.path.isdir(input_path):
        targets = [input_path]
    else:
        targets = [
            f for f in os.listdir(input_path) if not os.path.isdir(os.path.join(input_path, f))
        ]
        targets = [os.path.join(input_path, f) for f in targets]

    # load comparison images
    start = time.time()
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


    # compute reid features for comparison images
    args.first_reid = False
    features = reidentify(model=reid_model, targets=reid_targets, args=args, device=device, cfg=config, cutted_bboxes=[], features=[])

    # predictions array for later evaluation
    predictions = []
    all_predictions = []
    # for later visualization
    all_img_ids = []
    images_vis = {}
    # timing
    timing_pre_reid = []
    timing_detection = []
    timing_reidentification = []
    timing_postprocess = []


    # process the images
    for iter, image_path in enumerate(targets):

        _, ext = os.path.splitext(image_path)
        img_id = image_path.split('/')[-1].replace(ext, '')
        all_img_ids.append(img_id)
        # create 'window boxes' for the reidentification module
        boxes = []
        boxes_start = []

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = image.shape
        if args.visualize:
            images_vis[str(img_id)] = image.copy()

        mask_params = [0, 0]
        if args.crop_image_table:
            # area around the table is cutted out
            x1, y1, crop_w, crop_h = [550, 550, 1600, 1020]
            slice_y = slice(y1, y1+crop_h)
            slice_x = slice(x1, x1+crop_w)
            image = image[slice_y, slice_x, ...]
            height, width, _ = image.shape
            mask_params = [x1, y1]


        if args.reverse:
            # reidentification module
            args.first_reid = True
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            img = cv2.resize(image, (int(width/args.downscale_factor), int(height/args.downscale_factor)))
            boxes = [img]

            distmat = reidentify(model=reid_model1, cutted_bboxes=boxes, args=args, device=device, cfg=cfg, targets=[], features=features)
            end.record()
            torch.cuda.synchronize()
            timing_pre_reid.append((start.elapsed_time(end) / 1e3))
            # do postprocess
            # get the "best matching" feature vectors compared with targets of comparison images
            min_vals=[]
            for i in range(len(reid_targets)):
                below_t = np.argwhere(distmat[i] <= args.threshold_reid1)
                val = distmat[i][below_t].tolist()
                sorted_indices = np.argsort(list(chain(*val)))
                below_t = below_t[sorted_indices]
                top_k = below_t[:args.max_crops_reid1]
                top_k = list(chain(*top_k))
                min_vals += top_k

            # remove duplicates from list
            min_vals = list(set(min_vals))

            # get position for every feature vector (in full sized original image)
            dist = 16*args.downscale_factor  # distance between feature vectors in full sized image
            features_in_row = np.floor(width/dist)
            # get x and y coordinates
            y = np.floor(np.divide(min_vals, features_in_row)) * dist + dist/2
            x = np.remainder(min_vals, features_in_row) * dist + dist/2
            coordinates = list(zip(x,y))


            boxes = []
            box_size = args.box_size
            for (x, y) in coordinates:
                boxes.append([x,y,x+box_size,y+box_size])
            # positions close to each other often create bbox --> huge overlap 
            boxes_for_det = non_maximum_suppression(boxes, iou_threshold=args.iou_threshold_reid1)
            
            boxes_start = []
            boxes = []
            for box in boxes_for_det:
                # cut out box for detection
                x1, y1, crop_w, crop_h = [int(box[0]-(box_size/2)), int(box[1]-(box_size/2)), box_size, box_size]
                # check for image bounds
                box[0] = x1 = max(0, x1)
                box[1] = y1 = max(0, y1)
                box[2] = crop_w = min(crop_w, width-x1)
                box[3] = crop_h = min(crop_h, height-y1)
                slice_y = slice(y1, y1+crop_h)
                slice_x = slice(x1, x1+crop_w)
                image_crop = image[slice_y, slice_x, ...]
                boxes.append(image_crop)
                # parameter to compute global coordinates of predicted boxen in detection
                boxes_start.append([x1,y1,crop_w,crop_h])

            if args.visualize:
                img_vis = image.copy()
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                # visualize the crops(boxes) that will be analyzed in further process
                path_ = f'{args.output_path}/crops_for_det/'
                os.makedirs(path_, exist_ok=True)
                path_ = f'{args.output_path}/crops_for_det/{img_id}'
                for box in boxes_for_det:
                    img_vis = cv2.rectangle(img_vis, (int(box[0]), int(box[1])), (int(box[2])+int(box[0]), int(box[3])+int(box[1])), color=(0,0,255), thickness=4)
                cv2.imwrite(f'{path_}.png', img_vis)
            
        else:
            y_start = 0
            x_start = 0
            crop_height = args.crop_height
            # sliding window over image, check for image bounds
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


        # detection module
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        boxes, preds = detection_reverse(args, ort_sess=ort_sess, cropped_boxes=boxes, boxes_start=boxes_start, det_model=det_model, mask_params=mask_params)
        end.record()
        torch.cuda.synchronize()
        timing_detection.append((start.elapsed_time(end) / 1e3))


        # reidentification module
        args.first_reid = False
        start.record()
        distmat = reidentify(model=reid_model, cutted_bboxes=boxes, args=args, device=device, cfg=cfg, targets=[], features=features)
        end.record()
        torch.cuda.synchronize()
        timing_reidentification.append((start.elapsed_time(end) / 1e3))


        start.record()
        mins_ = []
         # get the indices of the n preds with "highest similarity"
        for i in range(len(reid_targets)):
            below_t = np.argwhere(distmat[i] <= args.threshold_reid2)
            val = distmat[i][below_t].tolist()
            sorted_indices = np.argsort(list(chain(*val)))
            below_t = below_t[sorted_indices]
            top_k = below_t[:args.max_boxes_reid2]
            mins_.append(top_k)
        

        # select bounding boxes with "highest similarity" after reidentification
        matched = set()
        for i, mins in enumerate(mins_):
            for j, ele in enumerate(mins):
                #score, indice = ele
                indice = ele.item()
                # check if prediction was already selected
                if indice not in matched:
                    matched.add(indice)
                    pred = {}
                    pred['image_id'] = int(img_id)
                    pred['category_id'] = 1
                    pred['score'] = preds[indice]['score']
                    b = preds[indice]['bbox'].copy()
                    # change bbox values to height, width
                    b[2] = b[2]-b[0]
                    b[3] = b[3]-b[1]
                    pred['bbox'] = b.tolist()
                    predictions.append(pred)
        end.record()
        torch.cuda.synchronize()
        timing_postprocess.append((start.elapsed_time(end) / 1e3))


        # just for evaluation
        # collect all predictions of detection module and get it in shape for evaluation
        if args.eval:
            for p in preds:
                p['image_id'] = int(img_id)
                p['category_id'] = 1
                b = p['bbox'].copy()
                # change bbox values to height, width
                b[2] = b[2]-b[0]
                b[3] = b[3]-b[1]
                p['bbox'] = b.tolist()
            # append formatted predictions of current image to all predictions
            all_predictions += preds


    if args.eval:
        # evaluation of boxes after detection
        gt_path = args.ground_truth_annotations
        os.makedirs(f'{args.output_path}/recall/', exist_ok=True)

        matched_dts = eval(gt_path, all_predictions, csv_path=f'{args.output_path}/recall/detection.csv')

        if args.visualize:
            # visualize predictions of detection (only those that matched with a ground truth)
            path_ = f'{args.output_path}/dets_after_detection/'
            visualize(matched_dts, path_, all_img_ids, all_predictions, images_vis, args, all=False)

        # evaluation of boxes after ReID2
        matched_dts = eval(gt_path, predictions, csv_path=f'{args.output_path}/recall/reid.csv')

        if args.visualize:
            # visualize only those boxes that matched with a ground truth - after reidentification
            path_ = f'{args.output_path}/dets_after_reid2_matched/'
            visualize(matched_dts, path_, all_img_ids, predictions, images_vis, args, all=False)
    
    if args.visualize:
            # visualize all boxes that remain after second reidentification
            path_ = f'{args.output_path}/dets_after_reid2/'
            visualize(matched_dts, path_, all_img_ids, predictions, images_vis, args, all=True)

    if args.reverse:
        print(f'Inferenzzeit für vorgelagert Wiedererkennung: {np.mean(timing_pre_reid)}')
    print(f'Inferenzzeit für Detektion: {np.mean(timing_detection)}')
    print(f'Inferenzzeit für nachgelagert Wiedererkennung: {np.mean(timing_reidentification)}')
    print(f'Inferenzzeit für Postprocess: {np.mean(timing_postprocess)}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
