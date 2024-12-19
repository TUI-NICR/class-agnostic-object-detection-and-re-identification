import argparse
from functools import wraps
import os
import cv2
import numpy as np
import onnxruntime as ort
import time

IKEA_META = [
    {'color': (70, 0, 129), 'name': 'table_top'},
    {'color': (0, 120, 220), 'name': 'leg'},
    {'color': (220, 100, 255), 'name': 'shelf'},
    {'color': (255, 231, 6), 'name': 'side_panel'},
    {'color': (130, 0, 89), 'name': 'front_panel'},
    {'color': (64, 221, 251), 'name': 'bottom_panel'},
    {'color': (255, 5, 5), 'name': 'rear_panel'}
]

ATTACH_META = [
    {'color': (70, 0, 129), 'name': 'Screwdriver'},
    {'color': (0, 120, 220), 'name': 'Leg'},
    {'color': (220, 100, 255), 'name': 'WallSpacerTop'},
    {'color': (255, 231, 6), 'name': 'ScrewNoHead'},
    {'color': (130, 0, 89), 'name': 'Wrench'},
    {'color': (64, 221, 251), 'name': 'Board'},
    {'color': (255, 5, 5), 'name': 'ScrewWithHead'},
    {'color': (0, 255, 255), 'name': 'ThreadedTupeFemale'},
    {'color': (0, 0, 255), 'name': 'ThreadedRod'},
    {'color': (0, 255, 0), 'name': 'Manual'},
    {'color': (255, 0, 0), 'name': 'Hammer'},
    {'color': (0, 252, 124), 'name': 'WallSpacerMounting'}
]

def parse_args():
    parser = argparse.ArgumentParser('inference with onnxruntime for Mask R-CNN')
    parser.add_argument(
        '--onnx-filepath',
        type=str,
        default='/path/to/ikea_asm_swin_tiny/2022_12_14-15_46_03-093460/deploy/end2end_optimized_extended.onnx',
        help='filepath to onnx model of Mask R-CNN',
    )
    parser.add_argument(
        '--img-or-video-filepath',
        type=str,
        default='/path/to/IKEA_ASM/ANU_ikea_dataset_images/Kallax_Shelf_Drawer/0001_black_table_02_01_2019_08_16_14_00/dev3/images/000000.jpg',
        help='filepath to input image/ video'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.3,
        help='only show instances with a confidence score higher than this threshold',
    )
    parser.add_argument(
        '--dataset',
        default='ikea',
        choices=('ikea', 'attach'),
        help='dataset on which the model was trained on'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='wait for N frames and process them together',
    )
    return parser.parse_args()


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        time_start = time.time()
        result = f(*args, **kw)
        time_end = time.time()
        time_diff = (time_end - time_start) * 1000
        print(f"function `{f.__name__}` took: {time_diff:.0f} milliseconds.")
        return result
    return wrap

# @timing
def preprocess_image(img_list):
    h, w, _ = img_list[0].shape

    # max_size = 896
    max_size = 768
    # max_size = 640

    scale = min(max_size / w, max_size / h)

    height_model = round(h * scale)
    width_model = round(w * scale)

    # both need to be divisible by 32
    height_model -= height_model % 32
    width_model -= width_model % 32

    # height_model = 608
    # width_model = 768

    # print(f"{height_model=}")
    # print(f"{width_model=}")

    imgs_batched = []
    for img in img_list:
        img = cv2.resize(img, (width_model, height_model))
        # keep image in BGR mode
        imgs_batched.append(img)

    imgs_batched = np.array(imgs_batched, dtype=np.float32)

    # normalize
    mean = np.float32([103.53, 116.28, 123.675])
    std = np.float32([57.375, 57.12, 58.395])
    imgs_batched -= mean
    imgs_batched /= std

    # swap dimensions to B,C,H,W for model
    imgs_batched = imgs_batched.transpose((0,3,1,2))

    return imgs_batched

# @timing
def postprocess(outputs, shape_orig, shape_pre):
    height, width = shape_orig[:2]
    height_pre, width_pre = shape_pre[-2:]
    width_scale = width / width_pre
    height_scale = height / height_pre

    dets, labels, masks = outputs

    return_list = []

    for idx in range(len(dets)):
        return_list.append([])
        for det, label, mask in zip(dets[idx], labels[idx], masks[idx]):
            score = det[-1]
            # threshold by score
            if score < args.confidence_threshold or score == 0:
                continue

            # resize box to full image size
            bbox = det[:4] * (width_scale, height_scale, width_scale, height_scale)
            bbox = bbox.astype(np.int32)

            # postprocess mask
            # make mask the same size as the image
            mask_fullsize = cv2.resize(mask, (width, height))

            # binarize mask
            mask_cropped = mask_fullsize[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            mask_cropped = np.where(mask_cropped > 0.5, 1, 0).astype(np.bool_)

            return_list[-1].append({
                'bbox': bbox, 'mask': mask_cropped, 'label': label, 'score': score
            })
    return return_list

@timing
def draw_outputs(img_list, outputs_postprocessed, meta):
    img_list_out = []
    for idx, img in enumerate(img_list):
        img_out = img.copy()
        for pred in outputs_postprocessed[idx]:
            bbox = pred['bbox']
            mask = pred['mask']
            label = pred['label']
            score = pred['score']

            # draw bounding box
            img_out = cv2.rectangle(
                img_out,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color=meta[label]['color']
            )

            # overlay colored mask on image
            bbox_slice = (slice(bbox[1], bbox[3]), slice(bbox[0], bbox[2]))
            img_out[bbox_slice][mask] = img_out[bbox_slice][mask] * 0.5 + 0.5 * np.array(meta[label]['color'])

            # write label and confidence score onto image
            cv2.putText(
                img_out,
                f"{meta[label]['name']} {score:.2f}",
                (bbox[0], bbox[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=meta[label]['color'],
                thickness=1
            )
        img_list_out.append(img_out.astype(np.uint8))
    return img_list_out


# @timing
def whole_inference(ort_sess, frame_list, meta):
    # preprocessing
    frames_pre = preprocess_image(frame_list)

    # @timing
    def run_onnx():
        outputs = ort_sess.run(
            output_names=['dets', 'labels', 'masks'],
            input_feed={'input': frames_pre},
        )
        return outputs
    outputs = run_onnx()

    # postprocessing
    outputs_postprocessed = postprocess(
        outputs,
        shape_orig=frame_list[0].shape,
        shape_pre=frames_pre[0].shape
    )

    # draw predictions
    frames_with_preds = draw_outputs(
        frame_list,
        outputs_postprocessed=outputs_postprocessed,
        meta=meta
    )
    return frames_with_preds


def main(args):
    # get appropriate meta data for colors and class names
    if args.dataset == 'ikea':
        meta = IKEA_META
    elif args.dataset == 'attach':
        meta = ATTACH_META
    else:
        raise ValueError(f"no meta data available for dataset {args.dataset}")

    # initialize onnxruntime session from provided filepath
    ort_sess = ort.InferenceSession(args.onnx_filepath, providers=['CUDAExecutionProvider'])

    img_or_video_file_ending = os.path.splitext(args.img_or_video_filepath)[1][1:]
    if img_or_video_file_ending in ['mp4', 'mov', 'avi', 'mkv']:
        cap = cv2.VideoCapture(args.img_or_video_filepath)
        counter = 0
        frame_list = []

        while cap.isOpened():
            # read a frame from the video
            ret, frame = cap.read()

            # if the frame was not read succesfully, break the loop
            if not ret:
                break

            counter += 1
            if counter % 60 != 0:
                continue

            # crop frame
            if args.dataset == 'attach':
                frame = frame[0:1440, 351:2159, :]
                # frame = frame[200:1440, 500:2159, :]

            frame_list.append(frame)

            if len(frame_list) == args.batch_size:

                # inference with pre- and postprocessing and drawing predictions
                frames_with_preds = whole_inference(ort_sess, frame_list, meta)

                frame_list = []

                # show result
                for frame in frames_with_preds:
                    cv2.imshow(args.img_or_video_filepath, cv2.resize(frame, None, fx=0.8, fy=0.8))
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
    else:
        # read image
        img_orig = cv2.imread(args.img_or_video_filepath)

        # inference with pre- and postprocessing and drawing predictions
        img_with_preds = whole_inference(ort_sess, [img_orig], meta)

        # show result
        cv2.imshow(args.img_or_video_filepath, cv2.resize(img_with_preds[0], None, fx=0.8, fy=0.8))
        cv2.waitKey(0)


if __name__ == '__main__':
    args = parse_args()
    main(args)
