import pandas as pd
import numpy as np
import os

SEED = 1337
rng = np.random.default_rng(SEED)

df = pd.read_csv(
    "redwood_reid_v1.csv", sep=",",
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

df_by_cls = df.groupby("class")

# split IDs 50/50 for each class between train and eval
test_ids = []
for class_, data in df_by_cls:
    data_id = data["id"].unique()
    test_choice = rng.choice(data_id, int(len(data_id)*0.5), replace=False)
    test_id = data[data["id"].isin(test_choice)]
    test_ids.append(test_id)
test_data = pd.concat(test_ids)
train_data = df.drop(test_data.index)
assert len(pd.merge(test_data, train_data, "inner")) == 0

# split images 80/20 for each ID in eval between query and galery
test_data_by_cls = test_data.groupby("class")
query_samples = []
for class_, data in test_data_by_cls:
    data_by_id = data.groupby("id")
    query_sample = data_by_id.sample(frac=0.2, random_state=int(rng.random()*1000))
    query_samples.append(query_sample)
query_data = pd.concat(query_samples)
test_data = test_data.drop(query_data.index)
assert len(pd.merge(test_data, query_data, "inner")) == 0

os.makedirs("redwood_reid_v1", exist_ok=True)
train_data.to_csv("redwood_reid_v1/train.csv", index=False)
test_data.to_csv("redwood_reid_v1/test.csv", index=False)
query_data.to_csv("redwood_reid_v1/query.csv", index=False)

print(train_data)
print(test_data)
print(query_data)
