import os
import cv2
import csv
import numpy as np
from tqdm import tqdm

PATH = "/path/to/Google_Scan"

# collect all available images (only 5 per ID) from thumbnails
data = [["path", "class", "id", "img_num", "ymax", "ymin", "xmax", "xmin"]]
for c_id, folder in enumerate(tqdm(os.listdir(PATH))):
    p = os.path.join(PATH, folder, "thumbnails")
    for i, img_name in enumerate(os.listdir(p)):
        img_p = os.path.join(p, img_name)
        img = cv2.imread(img_p)
        non_white = np.argwhere((img != (255, 255, 255)).all(axis=2))
        xmax = np.max(non_white[:, 1], axis=0)
        xmin = np.min(non_white[:, 1], axis=0)
        ymax = np.max(non_white[:, 0], axis=0)
        ymin = np.min(non_white[:, 0], axis=0)
        data.append([img_p, folder, c_id, i, ymax, ymin, xmax, xmin])

with open("gs_reid_v1.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(data)
