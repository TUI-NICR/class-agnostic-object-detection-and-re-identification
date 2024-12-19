import pandas as pd
import numpy as np
import os

SEED = 1337
rng = np.random.default_rng(SEED)

dataset_paths = [
    "../Attach/Attach_v1.csv",
    "../KTH/KTH_v1.csv",
    "../WorkingHands/WorkingHands_v1.csv"
]

dataset_names = [
    "attach_reid_v1",
    "kth_reid_v1",
    "workinghands_reid_v1"
]

dfs = []
for p in dataset_paths:
    df = pd.read_csv(
        p, sep=",",
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
    dfs.append(df)

# split images per ID 80/20 between gallery and query
for data, name in zip(dfs, dataset_names):
    data_by_cls = data.groupby("class")
    query_samples = []
    for class_, data_ in data_by_cls:
        data_by_id = data_.groupby("id")
        query_sample = data_by_id.sample(frac=0.2, random_state=int(rng.random()*1000))
        query_samples.append(query_sample)
    query_data = pd.concat(query_samples)
    test_data = data.drop(query_data.index)
    assert len(pd.merge(test_data, query_data, "inner")) == 0

    os.makedirs(name, exist_ok=True)
    test_data.to_csv(name + "/test.csv", index=False)
    query_data.to_csv(name + "/query.csv", index=False)
