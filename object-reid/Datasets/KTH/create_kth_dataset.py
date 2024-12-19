import os
import numpy as np
import xml.etree.ElementTree as ET
import csv

PATH = "/path/to/KTH"
SEED = 1337
rng = np.random.default_rng(SEED)

classes = {
    "hammer1": "hammer",
    "hammer2": "hammer",
    "hammer3": "hammer",
    "plier1": "pliers",
    "plier2": "pliers",
    "plier3": "pliers",
    "screw1": "screwdriver",
    "screw2": "screwdriver",
    "screw3": "screwdriver"
}

images = {
    "hammer1": [],
    "hammer2": [],
    "hammer3": [],
    "plier1": [],
    "plier2": [],
    "plier3": [],
    "screw1": [],
    "screw2": [],
    "screw3": []
}


def select(p):
    imgs = os.listdir(p)
    imgs = [img for img in imgs if "(1).jpg" not in img]
    ind = rng.choice(len(imgs))
    img = os.path.join(p, imgs[ind])
    return img


# Select one image from each combination of background, lighting, object and perspective
#  15 in total for each object
for bg in os.listdir(PATH):
    p = os.path.join(PATH, bg, "rgb")
    for lt in os.listdir(p):
        p = os.path.join(PATH, bg, "rgb", lt)
        for obj in os.listdir(p):
            p = os.path.join(PATH, bg, "rgb", lt, obj)
            content = os.listdir(p)
            if len(content) == 2:
                for cam in content:
                    p = os.path.join(PATH, bg, "rgb", lt, obj, cam)
                    img = select(p)
                    images[obj].append(img)
            else:
                img = select(p)
                images[obj].append(img)

header = ["path", "class", "id", "img_num", "ymax", "ymin", "xmax", "xmin"]
data = [header]
for id, (obj, paths) in enumerate(images.items()):
    for i, p in enumerate(paths):
        pb = p.replace("rgb", "bboxes").replace(".jpg", ".xml")
        if not os.path.exists(pb):
            pb = pb.replace("/Kinect/", "/kinect/")
        tree = ET.parse(pb)
        root = tree.getroot()
        box = root.find("object").find("bndbox")
        entry = [p, classes[obj], id, i, box.find("ymax").text, box.find("ymin").text, box.find("xmax").text, box.find("xmin").text]
        data.append(entry)

with open("KTH_v1.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(data)
