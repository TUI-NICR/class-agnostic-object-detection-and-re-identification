import csv
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

CSV_PATH = "/path/to/toDataset/attach_reid_v1"


stats = defaultdict(list)
for split in ["test", "query"]:
    with open(os.path.join(CSV_PATH, split + ".csv"), "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [r for r in reader]
    for e in tqdm(data):
        path, class_, id_, img_num, ymax, ymin, xmax, xmin = e
        with Image.open(path) as img:
            width, height = img.size
        stats[split].append({
            "img": {
                "img_width": width,
                "img_height": height
            },
            "bb": {
                "bb_ymax": int(ymax),
                "bb_ymin": int(ymin),
                "bb_xmax": int(xmax),
                "bb_xmin": int(xmin),
                "bb_width": int(xmax) - int(xmin),
                "bb_height": int(ymax) - int(ymin)
            },
            "object": {
                "class": class_,
                "id": int(id_),
                "img_num": int(img_num)
            }
        })
with open("attach_reid_v1_bb_stats.json", "w") as f:
    json.dump(stats, f)

#with open("attach_reid_v1_bb_stats.json", "r") as f:
#    stats = json.load(f)

stats = [s for d in stats.values() for s in d]

img_sizes = []
img_side_ratios = []
bb_sizes = []
bb_side_ratios = []
img_bb_area_ratios = []
img_bb_size_ratios = []

for s in stats:
    img_width = s["img"]["img_width"]
    img_height = s["img"]["img_height"]
    img_sizes.append((img_width, img_height))
    img_side_ratios.append(img_width / img_height)
    bb_width = s["bb"]["bb_width"]
    bb_height = s["bb"]["bb_height"]
    bb_sizes.append((bb_width, bb_height))
    bb_side_ratios.append(bb_width / bb_height)
    img_bb_area_ratios.append((img_width * img_height) / (bb_width * bb_height))
    img_bb_size_ratios.append((img_width / bb_width, img_height / bb_height))

img_sizes = list(zip(*img_sizes))
bb_sizes = list(zip(*bb_sizes))
img_bb_size_ratios = list(zip(*[(w, h) for w, h in img_bb_size_ratios if w < 16 and h < 16]))

fig, ax = plt.subplots(ncols=6, figsize=(35, 5), gridspec_kw=dict(left=0.016, right=1-0.016, top=1-0.1, bottom=0.1))

ax[0].hist2d(img_sizes[0], img_sizes[1], bins=20)
ax[0].plot(np.arange(1125), np.arange(1125)/9*16, linewidth=1, alpha=0.3)
ax[0].plot(np.arange(1500), np.arange(1500)/3*4, linewidth=1, alpha=0.3)
ax[0].plot(np.arange(1125)/9*16, np.arange(1125), linewidth=1, alpha=0.3)
ax[0].set_title("Image Sizes")

ax[1].hist(img_side_ratios, bins=20)
ax[1].set_title("Image Aspect Ratios")

ax[2].hist2d(bb_sizes[0], bb_sizes[1], bins=20)
ax[2].set_title("BB Sizes")

ax[3].hist(np.log2(bb_side_ratios), bins=20)
ax[3].set_xticks(ax[3].get_xticks(), np.exp2(ax[3].get_xticks()))
ax[3].set_title("BB Aspect Ratios")

ax[4].hist(np.log2(img_bb_area_ratios), bins=20)
ax[4].set_xticks(ax[4].get_xticks(), np.exp2(ax[4].get_xticks()))
ax[4].set_title("Image-BB Area Ratios")

ax[5].hist2d(np.log2(img_bb_size_ratios[0]), np.log2(img_bb_size_ratios[1]), bins=20)
ax[5].set_xticks(ax[5].get_xticks(), [f"{x:.2f}" for x in np.exp2(ax[5].get_xticks())])
ax[5].set_yticks(ax[5].get_yticks(), [f"{x:.2f}" for x in np.exp2(ax[5].get_yticks())])
ax[5].set_title("Image-BB Side Ratios")

fig.savefig("attach_v1_bb_props.pdf")
