import os
import json
import numpy as np
import csv

PATH = "/path/to/Attach/labels"
SEED = 420
rng = np.random.default_rng(SEED)

MIN_AREA = 2048

if False:
    images = {
        "Wrench": {},
        "Hammer": {},
        "Screwdriver": {}
    }

    # collect all images
    for t in os.listdir(PATH):
        if "front" in t:
            view = "front"
        elif "side" in t:
            view = "side"
        elif "spike" in t:
            view = "spike"
        else:
            continue
        p = os.path.join(PATH, t)
        labels = os.listdir(p)
        for lab in labels:
            p = os.path.join(PATH, t, lab)
            with open(p, "r") as f:
                data = json.load(f)
            img_p = data["imagePath"]
            for s in data["shapes"]:
                label = s["label"]
                if label in ["Wrench", "Hammer", "Screwdriver"]:
                    if view not in images[label]:
                        images[label][view] = {}
                    points = np.array(s["points"])
                    xmax = np.max(points[:, 0])
                    xmin = np.min(points[:, 0])
                    ymax = np.max(points[:, 1])
                    ymin = np.min(points[:, 1])
                    images[label][view][img_p] = (int(ymax), int(ymin), int(xmax), int(xmin))

    with open("TMP_attach_images.json", "w") as f:
        json.dump(images, f)

with open("TMP_attach_images.json", "r") as f:
    images = json.load(f)

with open("non_overlap_imgs.json", "r") as f:
    data = json.load(f)
allowed_imgs = {cls_: [img for imgs in vids.values() for img in imgs[1]] for cls_, vids in data.items()}

# Select 5 images for each view for each object, 15 in total
header = ["path", "class", "id", "img_num", "ymax", "ymin", "xmax", "xmin"]
data = [header]
nums = {}
for id, (cls, views) in enumerate(images.items()):
    if id not in nums:
        nums[id] = 0
    for v, imgs in views.items():
        imgs = list(imgs.items())
        imgs = [img for img in imgs if img[0] in allowed_imgs[cls]]
        imgs = [img for img in imgs if (img[1][0] - img[1][1]) * (img[1][2] - img[1][3]) >= MIN_AREA]
        inds = rng.choice(len(imgs), 5, replace=False)
        for ind in inds:
            img, (ymax, ymin, xmax, xmin) = imgs[ind]
            img = "/path/to/Attach/" + img.split("../")[-1].replace("tapes /", "tapes/").replace("tapes_tool_selection_0", "tapes")
            entry = [img, cls.lower(), id, nums[id], ymax, ymin, xmax, xmin]
            nums[id] += 1
            data.append(entry)

with open("Attach_v2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(data)
