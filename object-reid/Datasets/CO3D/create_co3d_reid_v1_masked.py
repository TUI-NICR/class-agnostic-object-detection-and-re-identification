import csv
import os

p = "co3d_reid_masked_v1"
os.makedirs(p, exist_ok=True)

# cut segmentation masks out of all images
for part in ["query", "test", "train"]:
    with open(f"co3d_reid_v1/{part}.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [r for r in reader]
    for i, e in enumerate(data):
        path = e[0]
        mask_path = path.replace("/images/", "/masks/").replace(".jpg", ".png")
        data[i] = [e[0], mask_path] + e[1:]
    header = [header[0], "mask_path"] + header[1:]
    with open(f"{p}/{part}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
