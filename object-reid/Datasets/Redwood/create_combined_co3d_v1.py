import os
import pandas as pd
import numpy as np

SEED = 1337
rng = np.random.default_rng(SEED)

rw_path = "/path/to/object-reid/Datasets/Redwood/redwood_reid_v1"
co3d_path = "/path/to/object-reid/Datasets/CO3D/co3d_reid_v1"

dest_path = "combined_redwood_co3d_reid_v1"

query_choice = {
    "co3d": [],
    "rw": []
}

for part in ["query", "test", "train"]:
    rw_data = pd.read_csv(
        os.path.join(rw_path, part+".csv"), sep=",",
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

    co3d_data = pd.read_csv(
        os.path.join(co3d_path, part+".csv"), sep=",",
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
    # Create split same size as CO3D but 50/50 from Redwood and CO3D
    # Take same percantage of IDs from each class.
    if part in ["query", "train"]:
        co3d_data_cls = co3d_data.groupby("class")
        co3d_keep = []
        for class_, data in co3d_data_cls:
            co3d_data_id = data["id"].unique()
            co3d_choice = rng.choice(co3d_data_id, int(len(co3d_data_id)*0.5), replace=False)
            if part == "query":
                query_choice["co3d"].append(co3d_choice)
            co3d_keep.append(data[data["id"].isin(co3d_choice)])
        co3d_keep = pd.concat(co3d_keep)

        target_ratio = len(co3d_keep) / len(rw_data)
        rw_data_cls = rw_data.groupby("class")
        rw_keep = []
        for class_, data in rw_data_cls:
            rw_data_id = data["id"].unique()
            rw_choice = rng.choice(rw_data_id, int(len(rw_data_id)*target_ratio), replace=False)
            if part == "query":
                query_choice["rw"].append(rw_choice)
            rw_keep.append(data[data["id"].isin(rw_choice)])
        rw_keep = pd.concat(rw_keep)

    # Make sure to include same IDs in gallery as in query split
    if part == "test":
        co3d_query_choice = np.concatenate(query_choice["co3d"])
        rw_query_choice = np.concatenate(query_choice["rw"])
        co3d_keep = co3d_data[co3d_data["id"].isin(co3d_query_choice)]
        rw_keep = rw_data[rw_data["id"].isin(rw_query_choice)]

    result = pd.concat([co3d_keep, rw_keep])
    result.to_csv(os.path.join(dest_path, part+".csv"), index=False)
