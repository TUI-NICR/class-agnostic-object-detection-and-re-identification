#%%
import numpy as np
import json
import cv2
import os
from pycocotools import mask as maskUtils
from nicr_mt_scene_analysis.visualization.instance import visualize_instance_pil
import argparse
from tqdm import tqdm
import pickle
import pycocotools

parser = argparse.ArgumentParser(
    description=(
        "convert masks from .json to .png"
    )
)

parser.add_argument(
    "--input-path",
    type=str,
    required=False,
    help="Path to mask in pkl format.",
)

parser.add_argument(
    "--output-path",
    type=str,
    required=False,
    help=(
        "Path where instance images should be stored"
    ),
)


def load_masks(ann):
    segm = ann#['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, ann['height'], ann['width'])
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, ann['size'][0], ann['size'][1])
    else:
        # rle
        rle = ann['counts']

    mask = maskUtils.decode(rle)

    return mask


def main() -> None:
    #input_path = args.input_path
    #output_path = args.output_path
    print("start converting...")
    objects = []
    with (open("/path/to/benchmark_rtmdet/hand/hand.pkl", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    with (open("/path/to/benchmark_rtmdet/h1/mask03.json", "r")) as best_ious_f:
        best_ious = json.load(best_ious_f)
    for i in range(20):
        id = objects[0][i]['img_id']
        x = objects[0][i]['pred_instances']['masks']
        scores = objects[0][i]['pred_instances']['scores']

        instance_img = np.zeros((x[0]['size'][0], x[0]['size'][1]),
                                                dtype=np.uint8)


        #id = file.split('/')[-1].split('.')[0]
        for i, ann in enumerate(x):
            if i in best_ious[str(id)]: #scores[i] > 0.2:
                    mask = maskUtils.decode(ann)
                    #mask = load_masks(ann)
                    if i == 0:
                        instance_img = mask.copy()
                    else:
                        # each instance gets a unique id
                        instance_img[mask > 0] = i + 1
            instance = instance_img
            instance = visualize_instance_pil(instance)
            output_path = '/path/to/benchmark_rtmdet/images/hand03'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            instance.save(output_path + str(id) + '.png')
    print("Done.")


if __name__ == "__main__":
    #args = parser.parse_args()
    main()

# %%
