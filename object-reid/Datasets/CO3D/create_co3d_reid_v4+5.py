import csv
import os

bad_classes = [
    "broccoli",
    "chair",
    "microwave",
    "motorcycle",
    "parkingmeter",
    "tv"
]

really_bad_classes = [
    "car",
    "stopsign",
]

os.makedirs("co3d_reid_v4", exist_ok=True)
os.makedirs("co3d_reid_v5", exist_ok=True)

# filter bad and really bad classes
for split in ["query", "test", "train"]:
    with open(f"co3d_reid_v1/{split}.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [r for r in reader]
    new_data_1 = [e for e in data if e[1] not in really_bad_classes]
    with open(f"co3d_reid_v4/{split}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(new_data_1)
    new_data_2 = [e for e in data if e[1] not in really_bad_classes + bad_classes]
    with open(f"co3d_reid_v5/{split}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(new_data_2)
