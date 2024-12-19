#!/usr/bin/env python
#SBATCH --gres=gpu:1
#SBATCH --nodelist=saturn12a
#SBATCH -p long
#SBATCH -t 4-0:0:0
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mona.koehler@tu-ilmenau.de

import argparse
import glob
import json
import os
import shutil
import tqdm

import cv2
from mmengine.config import Config
import numpy as np
from PIL import Image
from skimage import measure

from mmdet.apis import init_detector, inference_detector

# from visualize_rgb_from_json import COLORS
COLORS = [
    (129, 0, 70), (220, 120, 0), (255, 100, 220), (6, 231, 255), (89, 0, 130),
    (251, 221, 64), (5, 5, 255), (255, 255, 0), (255, 0, 0), (0, 255, 0),
    (0, 0, 255), (124, 252, 0)
]

IKEA_ASM_CATEGORIES = [
    {'id': 1, 'name': 'table_top'},
    {'id': 2, 'name': 'leg'},
    {'id': 3, 'name': 'shelf'},
    {'id': 4, 'name': 'side_panel'},
    {'id': 5, 'name': 'front_panel'},
    {'id': 6, 'name': 'bottom_panel'},
    {'id': 7, 'name': 'rear_panel'}
],
ATTACH_CATEGORIES = [
    {'id': 1, 'name': 'Screwdriver'},
    {'id': 2, 'name': 'Leg'},
    {'id': 3, 'name': 'WallSpacerTop'},
    {'id': 4, 'name': 'ScrewNoHead'},
    {'id': 5, 'name': 'Wrench'},
    {'id': 6, 'name': 'Board'},
    {'id': 7, 'name': 'ScrewWithHead'},
    {'id': 8, 'name': 'ThreadedTupeFemale'},
    {'id': 9, 'name': 'ThreadedRod'},
    {'id': 10, 'name': 'Manual'},
    {'id': 11, 'name': 'Hammer'},
    {'id': 12, 'name': 'WallSpacerMounting'}
 ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='ikea',
        choices=('ikea', 'attach'),
        help='dataset on which the model was trained on'
    )
    parser.add_argument(
        '--images-or-video-path',
        default='/path/to/IKEA_ASM/original/data/ANU_ikea_dataset_video',
        help='root path to images to predict on'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--checkpoint-filepath',
        type=str,
        default='work_dirs/ikea_swin_tiny/2022_12_14-15_46_03-093460/epoch_11.pth',
        help='name of the model in cfg.OUTPUT_DIR such as [ResNeXt.pth] ',
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default='configs/swin/ikea_swin_tiny.py',
        help='config to use',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='threshold for detection_heads',
    )
    parser.add_argument(
        '--output-path',
        default='/path/to/IKEA_ASM/predictions_SwinT_all_cameras',
        type=str,
        help='path where results are stored',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='show results'
    )
    parser.add_argument(
        '--start-index-video',
        type=int,
        default=0,
        help='Which index in video list to use for starting predictions. '
             'This is helpful when running the script multiple times in parallel.'
    )
    return parser.parse_args()


def results2json(results, score_threshold=0.1, image_id_start=0, segment_id_start=0):
    json_results = []
    segment_id = 0
    for idx, res in enumerate(results):
        label_list = res.pred_instances.labels.cpu().numpy()
        bbox_list = res.pred_instances.bboxes.cpu().numpy()
        segm_list = res.pred_instances.masks.cpu().numpy()
        score_list = res.pred_instances.scores.cpu().numpy()
        for i in range(len(res.pred_instances.labels)):
            label = label_list[i]
            bbox = bbox_list[i]
            segm = segm_list[i]
            score = score_list[i]
            if score < score_threshold:
                continue
            data = dict()
            data['image_id'] = idx + image_id_start
            data['id'] = segment_id + segment_id_start
            segment_id += 1
            data['bbox'] = bbox.tolist()    # xyxy format
            data['score'] = float(score)
            data['category_id'] = int(label) + 1
            segment = []
            contours = measure.find_contours(segm, 0.5)
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                segment.append(segmentation)
            data['segmentation'] = segment
            data['area'] = int(np.sum(segm))
            json_results.append(data)

    return json_results


def inference_and_update_json(
    model, batch, test_json, test_imgs, image_ids, threshold
):
    results = inference_detector(model, batch)
    res_json = results2json(
        results,
        score_threshold=threshold,
        image_id_start=image_ids[0],
        segment_id_start=len(test_json['annotations']),
    )
    test_json['annotations'].extend(res_json)

    for id, img in zip(image_ids, batch):
        if isinstance(img, np.ndarray):
            height, width, _ = img.shape
        elif isinstance(img, str):    # assuming it is a filepath
            with Image.open(img) as img_pil:
                width, height = img_pil.size
        test_json['images'].append({
            'id': id,
            'file_name': os.path.join(os.path.dirname(test_imgs[0]), str(id).zfill(6)),
            'height': height,
            'width': width
        })
    return test_json


def main():
    args = parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    cfg = Config.fromfile(args.cfg)
    cfg.dump(os.path.join(args.output_path, os.path.basename(args.cfg)))

    print("Load image/ video filepaths", flush=True)
    image_filepaths_dict = {}
    for root, dirs, files in os.walk(args.images_or_video_path):
        if os.path.split(root)[-1] == 'images':
            identifier = os.path.relpath(root, args.images_or_video_path).replace('/', '__')
            if identifier not in image_filepaths_dict:
                image_filepaths_dict[identifier] = []
        # for filename in files[:50]:    # TODO remove
        for filename in files:
            if filename.endswith(('jpg', 'png', 'avi', 'mp4')):
                if args.dataset == 'attach':
                    if filename.endswith(('jpg', 'png')):
                        identifier = root.split('/')[-2]
                    elif filename.endswith(('avi', 'mp4')):
                        identifier = os.path.relpath(root, args.images_or_video_path).replace('/', '__')
                    if identifier not in image_filepaths_dict:
                        image_filepaths_dict[identifier] = []
                image_filepaths_dict[identifier].append(os.path.join(root, filename))
        # if len(image_filepaths_dict) > 10:    # TODO remove
        #     break

    print("Build model", flush=True)
    model = init_detector(args.cfg, args.checkpoint_filepath, device='cuda:0')

    print("Copy checkpoint", flush=True)
    # copy checkpoint
    shutil.copy2(
        args.checkpoint_filepath,
        os.path.join(args.output_path, os.path.basename(args.checkpoint_filepath))
    )

    if args.dataset == 'ikea':
        categories = IKEA_ASM_CATEGORIES
    elif args.dataset == 'attach':
        categories = ATTACH_CATEGORIES

    print(f'using start index: {args.start_index_video}')
    for idx, (identifier, test_imgs) in enumerate(tqdm.tqdm(list(image_filepaths_dict.items())[args.start_index_video:])):
        print(f"\nProcessing {idx+1}/{len(image_filepaths_dict)-args.start_index_video} {identifier}", flush=True)

        out_json_fp = os.path.join(args.output_path, f'detections_{identifier}_thres_{args.threshold}.json')
        if os.path.exists(out_json_fp):
            continue

        test_json = {
            'categories': categories,
            'images': [],
            'annotations': [],
        }

        # reading from video
        if len(test_imgs) == 1 and test_imgs[0].endswith(('avi', 'mp4')):
            cap = cv2.VideoCapture(test_imgs[0])
            # Find the number of frames
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            count = 0
            batch = []
            image_ids = []
            while cap.isOpened():
                count += 1
                print(f"{count}/{video_length}", end="\r")
                # Extract the frame
                ret, frame = cap.read()
                if not ret:
                    continue
                batch.append(frame)
                image_ids.append(count)
                if len(batch) == args.batch_size:
                    test_json = inference_and_update_json(
                        model, batch, test_json, test_imgs, image_ids, args.threshold
                    )
                    batch = []
                    image_ids = []
                    # TODO remove!
                    # break

                # If there are no more frames left
                if (count > (video_length-1)):
                    if len(batch):
                        # Process final batch
                        test_json = inference_and_update_json(
                            model, batch, test_json, test_imgs, image_ids, args.threshold
                        )
                    # Release the feed
                    cap.release()
                    # Print stats
                    print ("Done extracting frames.\n%d frames extracted" % count)
                    break

        # processing separate frames
        else:
            test_imgs_batched = [
                test_imgs[i:i+args.batch_size]
                for i in range(0, len(test_imgs), args.batch_size)
            ]

            image_idx = 0
            for batch in test_imgs_batched:
                results = inference_detector(model, batch)
                res_json = results2json(
                    results,
                    score_threshold=args.threshold,
                    image_id_start=image_idx,
                    segment_id_start=len(test_json['annotations']),
                )
                test_json['annotations'].extend(res_json)

                for img_fp in batch:
                    with Image.open(img_fp) as im:
                        width, height = im.size

                    test_json['images'].append({
                        'id': image_idx,
                        'file_name': os.path.relpath(test_imgs[image_idx], args.images_or_video_path),
                        'height': height,
                        'width': width
                    })

                    image_idx += 1

        os.makedirs(args.output_path, exist_ok=True)
        with open(out_json_fp, 'w') as outfile:
            json.dump(test_json, outfile)
        print(f"Output json file '{out_json_fp}' created", flush=True)

        if args.show:
            id_to_category = {c['id']: c['name'] for c in categories}
            visualization_output_path = os.path.join(args.output_path, 'visualization')
            os.makedirs(visualization_output_path, exist_ok=True)

            for image_meta in test_json['images']:
                img_id = image_meta['id']
                file_name = image_meta['file_name']
                annos = [anno for anno in test_json['annotations'] if anno['image_id'] == img_id]
                img = cv2.imread(os.path.join(args.images_or_video_path, file_name))
                img_orig = img.copy()

                for anno in annos:
                    bbox = anno['bbox']
                    contours = []
                    for contour in anno['segmentation']:
                        id = 0
                        cnt = len(contour)
                        c = np.zeros((int(cnt / 2), 1, 2), dtype=np.int32)
                        for j in range(0, cnt, 2):
                            c[id, 0, 0] = contour[j]
                            c[id, 0, 1] = contour[j + 1]
                            id = id + 1
                        contours.append(c)
                    color_cat = COLORS[anno['category_id'] -1]
                    cv2.drawContours(img, contours, -1, color_cat, -1)
                    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_cat, 1)
                    x1, y1 = bbox[:2]
                    category_name = id_to_category[anno['category_id']]
                    score = anno['score']
                    text = f'{category_name} {score:.3f}'
                    cv2.putText(img, text, (int(x1) - 10, int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 1)
                img = (img * 0.6 + img_orig * 0.4).astype(np.uint8)
                im_path = os.path.join(visualization_output_path, file_name.replace('/', '__'))
                cv2.imwrite(im_path, img)
                print(f"written {im_path}")


if __name__ == '__main__':
    main()
