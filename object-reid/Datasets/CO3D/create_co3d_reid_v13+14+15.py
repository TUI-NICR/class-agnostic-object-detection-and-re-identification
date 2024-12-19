import numpy as np
import pandas as pd
import os

SEED = 1337
rng = np.random.default_rng(SEED)

for split in ["train", "query", "test"]:
    df_100 = pd.read_csv(
        f"../../Object-ReID/toDataset/co3d_reid_v10/{split}.csv", sep=",",
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

    data_class = df_100["class"].unique()
    size_100 = len(df_100)

    # pick 50% of classes at random to discard and determine size of new dataset
    if split == "train":
        choice_050 = rng.choice(data_class, int(len(data_class)*0.50), replace=False)
    df_050 = df_100[df_100["class"].isin(choice_050) == False]
    size_050 = len(df_050)

    # re-add 50% of discarded classes and determine size of new dataset
    if split == "train":
        choice_025 = rng.choice(choice_050, int(len(choice_050)*0.5), replace=False)
    df_075 = df_100[df_100["class"].isin(choice_025) == False]
    size_075 = len(df_075)

    # remove IDs unti alld datasets have same number of IDs as dataset with 50% of classes
    if split in ["train", "query"]:
        dfs = [df_075, df_100]
        for i, (df_, size) in enumerate(zip(dfs, [size_075, size_100])):
            df_cls = df_.groupby("class")
            samples = []
            for class_, data in df_cls:
                data_id = data["id"].unique()
                choice = rng.choice(data_id, int(len(data_id)*(size_050/size)), replace=False)
                sample = data[data["id"].isin(choice)]
                samples.append(sample)
            dfs[i] = pd.concat(samples)
        df_075, df_100 = dfs
    elif split == "test":
        query_id_075 = dfs[0]["id"].unique()
        df_075 = df_075[df_075["id"].isin(query_id_075)]
        query_id_100 = dfs[1]["id"].unique()
        df_100 = df_100[df_100["id"].isin(query_id_100)]

    os.makedirs("co3d_reid_v13", exist_ok=True)
    df_100.to_csv(f"co3d_reid_v13/{split}.csv", index=False)
    os.makedirs("co3d_reid_v14", exist_ok=True)
    df_075.to_csv(f"co3d_reid_v14/{split}.csv", index=False)
    os.makedirs("co3d_reid_v15", exist_ok=True)
    df_050.to_csv(f"co3d_reid_v15/{split}.csv", index=False)
