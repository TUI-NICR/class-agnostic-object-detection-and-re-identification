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


def main() -> None:
    #input_path = args.input_path
    #output_path = args.output_path
    rgb_path = '/path/to/attach_benchmark/imgs_cropped/table/'
    print("start converting...")
    objects = []
    with (open("/path/to/benchmark_rtmdet/table/models/m.pkl", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    with (open("/path/to/benchmark_rtmdet/table/models/m.json", "r")) as best_ious_f:
        best_ious = json.load(best_ious_f)
    for i, rgb in enumerate(sorted(os.listdir(rgb_path))):
        id = objects[0][i]['img_id']
        x = objects[0][i]['pred_instances']['bboxes']
        rgb = str(id) + '.png'
        scores = objects[0][i]['pred_instances']['scores']
        rgb_img = os.path.join(rgb_path, rgb)
        bbox_image = cv2.imread(rgb_img)


        if len(best_ious[str(id)]) > 0:
                n = len(best_ious[str(id)])

                def bitget(byteval, idx):
                    return (byteval & (1 << idx)) != 0

                cmap = []
                for i in range(n):
                    r = g = b = 0
                    c = i
                    for j in range(8):
                        r = r | (bitget(c, 0) << 7-j)
                        g = g | (bitget(c, 1) << 7-j)
                        b = b | (bitget(c, 2) << 7-j)
                        c = c >> 3
                    cmap.append((r, g, b))
        y = 0
        for i, ann in enumerate(x):
            if i in best_ious[str(id)]:
                # draw bbox
                bbox = x[i]
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = bbox[2]
                y_max = bbox[3]
                colour = cmap[y]
                y+=1
                bbox_image = cv2.rectangle(bbox_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), colour, 2)

        output_path = '/path/to/benchmark_rtmdet/table/m/bboxes'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path+str(id)+'.png', bbox_image)

if __name__ == "__main__":
    #args = parser.parse_args()
    main()
# %%
