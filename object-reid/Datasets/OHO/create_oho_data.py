import numpy as np
import cv2
import os
import csv
from tqdm import tqdm

PATH = "/path/to/OHO_data/SONARO/SONARO_Masks_v2-3"
DEST = "/path/to/OHO"
HEAD = ["path", "mask_path", "class", "id", "img_num", "ymax", "ymin", "xmax", "xmin"]

# collect all OHO images and segmentation masks
data = []
c_mask_error = 0
for id_, cls_ in enumerate(tqdm(os.listdir(PATH))):
    p = os.path.join(PATH, cls_)
    for img_num, img_d in enumerate(os.listdir(p)):
        p = os.path.join(PATH, cls_, img_d)
        img_p = os.path.join(p, "masked_img.png")
        img = cv2.imread(img_p)
        mask = np.load(os.path.join(p, "masks.npy"))[..., 0]
        mask_inds = np.argwhere(mask > 0)
        if mask_inds.size == 0:
            c_mask_error += 1
            continue
        xmax = np.max(mask_inds[:, 1], axis=0)
        xmin = np.min(mask_inds[:, 1], axis=0)
        ymax = np.max(mask_inds[:, 0], axis=0)
        ymin = np.min(mask_inds[:, 0], axis=0)
        mask_dir = os.path.join(DEST, cls_, img_d)
        os.makedirs(mask_dir, exist_ok=True)
        mask_p = os.path.join(mask_dir, "mask.png")
        cv2.imwrite(mask_p, mask)
        data.append([img_p, mask_p, cls_, id_, img_num, ymax, ymin, xmax, xmin])

print("Faulty masks:", c_mask_error)
with open("oho_reid.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(HEAD)
    writer.writerows(data)
