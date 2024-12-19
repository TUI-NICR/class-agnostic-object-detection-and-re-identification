import pandas as pd
import numpy as np
import os

SEED = 1337
rng = np.random.default_rng(SEED)

df = pd.read_csv(
    "gs_reid_v1.csv",
    sep=",",
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

# split images 80/20 between galery and query for each ID
data_by_cls = df.groupby("class")
query_samples = []
for class_, data_ in data_by_cls:
    data_by_id = data_.groupby("id")
    query_sample = data_by_id.sample(frac=0.2, random_state=int(rng.random()*1000))
    query_samples.append(query_sample)
query_data = pd.concat(query_samples)
test_data = df.drop(query_data.index)
assert len(pd.merge(test_data, query_data, "inner")) == 0

name = "gs_reid_v1"
os.makedirs(name, exist_ok=True)
test_data.to_csv(name + "/test.csv", index=False)
query_data.to_csv(name + "/query.csv", index=False)
