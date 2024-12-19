import numpy as np
import pandas as pd
import os


SEED = 1337
rng = np.random.default_rng(SEED)

PATH = "oho_reid.csv"
NAME = "oho_reid_v1"

df = pd.read_csv(
    PATH, sep=",",
    dtype={
        "path": str,
        "mask_path": str,
        "class": str,
        "id": int,
        "img_num": int,
        "xmax": int,
        "xmin": int,
        "ymax": int,
        "ymin": int
    }
)

# sample 20 images for each ID
# split images 80/20 between galery and query
data_by_cls = df.groupby("class")
query_samples = []
test_samples = []
for class_, data in data_by_cls:
    sample = data.sample(n=20, random_state=int(rng.random()*1000))
    query_sample = sample.sample(frac=0.2, random_state=int(rng.random()*1000))
    test_sample = sample.drop(query_sample.index)
    query_samples.append(query_sample)
    test_samples.append(test_sample)
query_data = pd.concat(query_samples)
test_data = pd.concat(test_samples)
assert len(pd.merge(test_data, query_data, "inner")) == 0

os.makedirs(NAME, exist_ok=True)
test_data.to_csv(NAME + "/test.csv", index=False)
query_data.to_csv(NAME + "/query.csv", index=False)
