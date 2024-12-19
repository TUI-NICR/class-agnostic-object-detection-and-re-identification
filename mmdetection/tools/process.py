import argparse
from functools import wraps
import os
import cv2
import numpy as np
import onnxruntime as ort
import time


def parse_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        '--onnx-filepath',
        type=str,
        default='/path/to/rtmdet_deploy/test/2/end2end.onnx',
        help='filepath to onnx model of Mask R-CNN',
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
        default=0.50,
        help='delete all boxes with iou > threshold',
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='/path/to/rtmdet_deploy/bboxes/',
        help='filepath where preds should be stored'
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=320,
        help="crop height",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=320,
        help="crop width",
    )
    return parser.parse_args()


def draw_bboxes(image, pred_list, threshold):
    """
    draws the bounding boxes in the image with a unique colour for each box

    :param image: the rgb input image
    :param pred_list: list of predicted bounding boxes
    :param threshold: boxes with lower score than threshold will not be drawn
    :return: image with predicted bounding boxes
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

    img_out = image
    # iterate through all predictions
    for i, pred in enumerate(pred_list):
        bbox = pred['bbox']
        score = pred['score']
        # exclude bboxes with score lower than a predefined threshold
        if score > threshold:
            color = cmap[i]
            # draw the bbox into the image
            img_out = cv2.rectangle(
                img_out,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color = color, thickness=2
            )
    return img_out


def postprocess(outputs, confidence_threshold):
    """
    postprocess the predicted outputs, remove everything taht is unnecessary

    :param outputs: predictions of the model
    :return: list of the postprocessed outputs
    """
    dets, labels, masks = outputs

    pred_list = []
    for idx in range(len(dets)):
        pred_list.append([])
        # iterate thorugh detections of the model
        for det, label, mask in zip(dets[idx], labels[idx], masks[idx]):
            score = det[-1]
            bbox = det[:4]  # * (width_scale, height_scale, width_scale, height_scale)
            bbox = bbox.astype(np.int32)
            # append bboxes with corresponding scores and labels to the prediction list
            # append bbox only if score is high enough
            if score >= confidence_threshold:
                pred_list[-1].append({
                    'bbox': bbox, 'label': label, 'score': score
                })
    return pred_list


def compute_global_coordinates(pred_list, x_start, y_start, crop_width, crop_height):
    """
    compute global coordinates for the predicted bounding boxes

    :param pred_list: list with all predictions
    :param x_start: start x-coordinate of current crop
    :param y_start: start y-coordinate of current crop
    :param crop_width: width of one image crop
    :return: bounding boxes with global coordinates
    """
    bboxes_ = []
    for i, element in enumerate(pred_list):
        bbox_coordinates = element['bbox']
        if bbox_coordinates[0] != 0 and bbox_coordinates[1] != 0 and bbox_coordinates[2] != crop_width-1:
            # add the x and y starting positons to the predicted
            # edges of the bbox to receive global coordinates
            bbox_coordinates[0] += x_start
            bbox_coordinates[1] += y_start
            bbox_coordinates[2] += x_start
            bbox_coordinates[3] += y_start
            element['bbox'] = bbox_coordinates
            bboxes_.append(element)

    return bboxes_


def select_preds(preds, last_preds, iou_t):
    """
    removes "duplicate" predictions, compares predictions of current cop with the last crop

    :param preds: predictions of current crop
    :param last_pred: predictions of last crop
    :param iou_t: threshold for IoU
    :return: selected predictions
    """
    preds_ = []
    safed = []
    for i in range(len(preds)):
        predA = preds[i]
        highest_iou = 0
        boxA = predA['bbox']
        add = True
        for j in range(0, len(preds)):
            if j == i:
                continue
            predB = preds[j]
            boxB = predB['bbox']
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            # compute the area of intersection rectangle
            interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
            if interArea == 0:
                iou = 0
            # compute the area of both the prediction and ground-truth
            # rectangles
            else:
                boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
                boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)
            if iou > highest_iou:
                # new heighest IoU
                highest_iou = iou

            if j in safed:
                add = False
        if highest_iou < iou_t or add:
            preds_.append(predA)
            safed.append(i)

    return preds_


def main(args):
    # params
    # path to the already exported onnx model
    onnx_filepath = args.onnx_filepath
    # path to the input image or a directory with images 
    input_path = args.img_filepath
    # path where the results should be stored
    output_path = args.output_path
    # threshold for predictions, only bboxes with a higher prediction score will be visualized
    threshold = args.confidence_threshold
    # threshold for predicted boxes, removes "duplicates"
    iou_threshold = args.iou_threshold
    # whether the image as a whole should get processed too
    full_image = False
    # whether image crops should be processed or not
    crop_img = True
    # whether only a mask of the full image gets processed or everything
    mask_image = True

    # create directory where output images should be stored
    os.makedirs(output_path, exist_ok=True)

    # create the Inference session with the exported ONNX Model
    ort_sess = ort.InferenceSession(onnx_filepath, providers=['CUDAExecutionProvider'])

    # collect all iamges that are going to be processed in a list
    if not os.path.isdir(input_path):
        targets = [input_path]
    else:
        targets = [
            f for f in os.listdir(input_path) if not os.path.isdir(os.path.join(input_path, f))
        ]
        targets = [os.path.join(input_path, f) for f in targets]

    # process the images
    for t in targets:
        print(f"Processing '{t}'...")
        # get image id
        _, ext = os.path.splitext(t)
        img_id = t.split('/')[-1].replace(ext, '')
        # read input image
        image = cv2.imread(t)
        height, width, _ = image.shape

        # process image as a whole
        if full_image:
            # preprocess image
            imgs_batched = []
            imgs_batched.append(image)
            imgs_batched = np.array(imgs_batched, dtype=np.float32)
            imgs_batched = imgs_batched.transpose((0, 3, 1, 2))
            # process image through model
            outputs = ort_sess.run(
                        output_names=['dets', 'labels', 'masks'],
                        input_feed={'input': imgs_batched},
                    )
            pred_list = postprocess(outputs)
            # draw predicted bboxes
            img_out = draw_bboxes(image, pred_list, threshold)
            os.makedirs(f'{output_path}{img_id}/', exist_ok=True)
            # save image
            cv2.imwrite(f'{output_path}{img_id}/full.png', img_out)

        start1 = time.time()
        test = []
        # aply "mask" for image (only masked parts gets processed)
        if mask_image:
            x1 = 550
            y1 = 450
            crop_w = 1600
            crop_h = 1020
            slice_y = slice(y1, y1+crop_h)
            slice_x = slice(x1, x1+crop_w)
            image = image[slice_y, slice_x, ...]
            height, width, _ = image.shape
        if crop_img:
            all_preds = []
            os.makedirs(f'{output_path}{img_id}/', exist_ok=True)
            # define starting positon for the first image crop
            y_start = 0
            x_start = 0
            crop_height = args.crop_height
            # sliding window over image, check for image bounds
            # TODO: change second condition?
            while (y_start < height and crop_height >= (args.crop_height/2)):
                x_start = 0
                crop_width = args.crop_width
                last_preds = []
                while (x_start < width and crop_width >= (args.crop_width/2)):
                    # determine next crop
                    slice_y = slice(y_start, y_start+crop_height)
                    slice_x = slice(x_start, x_start+crop_width)
                    crop_image = image[slice_y, slice_x, ...]

                    # preprocess image
                    imgs_batched = []
                    imgs_batched.append(crop_image)
                    imgs_batched = np.array(imgs_batched, dtype=np.float32)
                    imgs_batched = imgs_batched.transpose((0, 3, 1, 2))
                    # process image through model
                    outputs = ort_sess.run(
                                output_names=['dets', 'labels', 'masks'],
                                input_feed={'input': imgs_batched},
                            )

                    pred_list = postprocess(outputs, threshold)

                    # compute global coordinates for bboxes

                    preds = compute_global_coordinates(pred_list[0], x_start, y_start, crop_width, crop_height)

                    # remove "duplicates"?
                    preds = select_preds(preds, last_preds, iou_threshold)
                    all_preds += preds
                    last_preds = preds

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

            # draw the predicted bboxes
            img_out = draw_bboxes(image, all_preds, threshold)
            # save the resulting image
            cv2.imwrite(f'{output_path}{img_id}/crop_all_2.png', img_out)
        test.append(len(all_preds))
        end1 = time.time()
        inference_time = end1 - start1
        print(f"Inference time: {inference_time} seconds")
    print(np.mean(test))


if __name__ == '__main__':
    # parse args
    args = parse_args()
    main(args)
