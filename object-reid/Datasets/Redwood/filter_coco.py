import json
import csv


with open("coco_class_map.json", "r") as f:
    coco_classes = json.load(f)

# filter coco classes from Redwood dataset
split_names = ["train", "test", "query"]
for name in split_names:
    with open(f"redwood_reid_v1/{name}.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [r for r in reader]
    mod_split = [header]
    for e in data:
        if e[1] in coco_classes:
            mod_split.append(e)
    with open(f"redwood_reid_coco/{name}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(mod_split)
    print(name, len(mod_split))
