import json
import cv2
import numpy as np

with open("/path/to/Attach/labels/labels_train_2023_04_24.json", "r") as f:
    data = json.load(f)

images = data["images"]
ann = data["annotations"]
cat = data["categories"]
cat = {x["id"]: x for x in cat}

exclude_cat = set(["Board", "Leg"])

example_p = "tapes/40__2__shuttleFront/e4sm_tape-2021-07-06_15-18-34.793831_png.shuttleFront/shuttleFront_Kinect_ColorImage1625577561514287000.jpg"
example_id = [x["id"] for x in images if x["file_name"] == example_p][0]

example_ann = []
for a in ann:
    if a["image_id"] == example_id:
        example_ann.append(a)

image = cv2.imread("/path/to/Attach/" + example_p)
for a in example_ann:
    box = np.array(a["bbox"], dtype=np.int32)
    cls_ = cat[a["category_id"]]["name"]
    if cls_ in exclude_cat:
        continue
    image = cv2.rectangle(image, box[:2], box[:2]+box[2:], (0, 0, 128), 2)
    image = cv2.putText(image, cls_, box[:2], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 128), 2)
image = cv2.resize(image, (1600, 900))

cv2.imshow("example", image)
cv2.waitKey(0)
