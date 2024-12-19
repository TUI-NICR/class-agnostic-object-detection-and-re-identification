import numpy as np
import pandas as pd
import os


SEED = 1337
rng = np.random.default_rng(SEED)

all_samples = {}

for split in ["train", "query", "test"]:
    df = pd.read_csv(
        f"/path/to/object-reid/Object-ReID/toDataset/co3d_reid_v10/{split}.csv", sep=",",
        dtype={
            "path": str,
            "class": str,
            "id": int,
            "img_num": int,
            "xmax": int,
            "xmin": int,
            "ymax": int,
            "ymin": int
        }
    )
    # reduce entire dataset by 90% by sampling 10% of IDs for each class
    if split in ["train", "query"]:
        df_by_cls = df.groupby("class")
        samples = []
        for class_, data in df_by_cls:
            data_id = data["id"].unique()
            choice = rng.choice(data_id, int(len(data_id)*0.1), replace=False)
            sample = data[data["id"].isin(choice)]
            samples.append(sample)
        all_samples[split] = pd.concat(samples)
    # ensure galery IDs match query
    elif split == "test":
        query_id = all_samples["query"]["id"].unique()
        samples = df[df["id"].isin(query_id)]
        all_samples[split] = samples

os.makedirs("co3d_reid_v12", exist_ok=True)
all_samples["train"].to_csv("co3d_reid_v12/train.csv", index=False)
all_samples["test"].to_csv("co3d_reid_v12/test.csv", index=False)
all_samples["query"].to_csv("co3d_reid_v12/query.csv", index=False)

print("train", all_samples["train"])
print("test", all_samples["test"])
print("query", all_samples["query"])
