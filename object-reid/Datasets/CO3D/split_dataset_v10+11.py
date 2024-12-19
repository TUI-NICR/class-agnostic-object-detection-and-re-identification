import pandas as pd
import numpy as np
import os

SEED = 1337
rng = np.random.default_rng(SEED)

for name, test_cls in [["10", True], ["11", False]]:
    df = pd.read_csv(
        f"co3d_reid_v10.csv", sep=",",
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
    if test_cls:
        test_classes = [
            "banana",
            "kite",
            "parkingmeter",
            "skateboard"
        ]
        # add test_classes to eval splits
        test_data = df[df["class"].isin(test_classes)]
        test_ids = [test_data]
    else:
        test_classes = []
        test_ids = []

    # split IDs 50/50 between eval and train parts for each non-test_classes class
    for class_, data in df_by_cls:
        if class_ in test_classes:
            continue
        data_id = data["id"].unique()
        test_choice = rng.choice(data_id, int(len(data_id)*0.5), replace=False)
        test_id = data[data["id"].isin(test_choice)]
        test_ids.append(test_id)
    test_data = pd.concat(test_ids)
    train_data = df.drop(test_data.index)
    assert len(pd.merge(test_data, train_data, "inner")) == 0

    # split eval images 80/20 for each ID into galery and query
    test_data_by_cls = test_data.groupby("class")
    query_samples = []
    for class_, data in test_data_by_cls:
        data_by_id = data.groupby("id")
        query_sample = data_by_id.sample(frac=0.2, random_state=int(rng.random()*1000))
        query_samples.append(query_sample)
    query_data = pd.concat(query_samples)
    test_data = test_data.drop(query_data.index)
    assert len(pd.merge(test_data, query_data, "inner")) == 0

    os.makedirs(f"co3d_reid_v{name}", exist_ok=True)
    train_data.to_csv(f"co3d_reid_v{name}/train.csv", index=False)
    test_data.to_csv(f"co3d_reid_v{name}/test.csv", index=False)
    query_data.to_csv(f"co3d_reid_v{name}/query.csv", index=False)

    print(train_data)
    print(test_data)
    print(query_data)
