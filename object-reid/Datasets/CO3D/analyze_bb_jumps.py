import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


#if False:
with open("TMP_co3d_consec_ious.json", "r") as f:
    ious = json.load(f)

df_orig = pd.DataFrame.from_dict({
    (i, j, int(k)): ious[i][j][k]
        for i in ious.keys()
            for j in ious[i].keys()
                for k in ious[i][j].keys()
    },
    orient="index"
)
df_orig = df_orig.rename_axis(["class", "id", "num"])

results = defaultdict(lambda: defaultdict(list))
results_with_id = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for iou_threshold in range(20):
    iou_threshold *= 0.05
    df_ = df_orig.copy()
    df_["iou"] = np.where(df_["iou"] > iou_threshold, True, False)
    for class_, df_class in df_.groupby("class"):
        for id_, df_id in df_class.groupby("id"):
            inds = np.squeeze(np.argwhere(df_id["iou"] == False), axis=1)
            if len(inds) > 0:
                lengths = np.concatenate((inds[:1], inds[1:] - inds[:-1], len(df_id["iou"]) - inds[-1:]))
                results[iou_threshold][class_].append(lengths)
                results_with_id[iou_threshold][class_][id_].append(lengths)
            else:
                results[iou_threshold][class_].append(np.array([len(df_id["iou"])]))
                results_with_id[iou_threshold][class_][id_].append(np.array([len(df_id["iou"])]))

df = pd.DataFrame.from_dict({
    (i, j, k): {"length": e}
        for i in results.keys()
            for j in results[i].keys()
                for k, e in enumerate(np.concatenate(results[i][j]))
    },
    orient="index"
)
df = df.rename_axis(["iou_threshold", "class", "i"])


ax = df.hist(by="iou_threshold", bins=50, figsize=(20, 20))
plt.savefig("sequence_length_by_threshold.pdf")
plt.close()

df = df.reset_index("i").groupby(["iou_threshold", "class", "length"]).count().reset_index("length")
df["i"] *= df["length"]
df = df.reset_index()
df["iou_threshold"] = df["iou_threshold"].round(2)

df = pd.pivot_table(
    data=df,
    index=["class", "length"],
    columns=["iou_threshold"],
    values="i"
)

df.to_pickle("TMP_DF.pkl")

fig, axes = plt.subplots(8, 7, figsize=(320, 210), layout="constrained")
i = 0
j = 0
for class_, data in df.groupby("class"):
    ax = data.plot(ax=axes[j][i], rot=90, kind="bar", stacked=True, legend=False, fontsize=8, title=class_, xticks=np.arange(0, len(data.index), 20), xlabel="length", figsize=(40, 30))
    i += 1
    if i == 7:
        i = 0
        j += 1
fig.savefig("sequence_length_by_threshold_scaled.svg")


#df = pd.read_pickle("TMP_DF.pkl")
df = df[[0.0, 0.4]]
fig, axes = plt.subplots(8, 7, figsize=(320, 210), layout="constrained")
i = 0
j = 0
for class_, data in df.groupby("class"):
    ax = data.plot(ax=axes[j][i], rot=90, kind="bar", stacked=False, legend=True, fontsize=8, title=class_, xticks=np.arange(0, len(data.index), 20), xlabel="length", figsize=(40, 30))
    i += 1
    if i == 7:
        i = 0
        j += 1
fig.savefig("sequence_length_by_threshold_0-0_and_0-4_scaled.svg")


results = results_with_id

df = pd.DataFrame.from_dict({
    (i, j, k, m): {"length": e}
        for i in results.keys()
            for j in results[i].keys()
                for k in results[i][j].keys()
                    for m, e in enumerate(np.concatenate(results[i][j][k]))
    },
    orient="index"
)
df = df.rename_axis(["iou_threshold", "class", "id", "i"])

df.to_pickle("TMP_DF_2.pkl")
#df = pd.read_pickle("TMP_DF_2.pkl")

df = df.reset_index("i")
df = df.groupby(["iou_threshold", "class", "id", "length"]).count()
df = df.reset_index()
df["iou_threshold"] = df["iou_threshold"].round(2)
df = df.sort_values(["iou_threshold", "class", "id", "length"])
df["i"] *= df["length"]
df["i"] = df.groupby(["iou_threshold", "class", "id"]).cumsum()["i"]
maxes = df.groupby(["iou_threshold", "class", "id"]).transform("max")["i"]
df["i"] /= maxes

df = pd.pivot_table(
    data=df,
    index=["iou_threshold", "class", "id"],
    columns=["length"],
    values = "i"
).ffill(axis=1).fillna(0)
aggs = [
    lambda x: x.quantile(0.20),
    lambda x: x.quantile(0.40),
    lambda x: x.quantile(0.60),
    lambda x: x.quantile(0.80)
]
df = df.groupby(["iou_threshold", "class"]).agg(aggs)
df = df.rename_axis(["length", "agg"], axis=1)

for iou_v in [0.0, 0.4, 0.95]:
    df_ = df.loc[iou_v]

    fig, axes = plt.subplots(8, 7, figsize=(240, 240), layout="constrained")
    axes = axes.flatten()
    for i, (class_, data) in enumerate(df_.groupby("class")):
        data = data.swaplevel(axis=1)
        for r in [f"<lambda_{k}>" for k in range(len(aggs))]:
            ax = data[r].T[class_].plot(ax=axes[i], legend=False, rot=90, kind="line", fontsize=8, title=class_, figsize=(30, 30))

    plt.savefig(f"lost_percentage_iou_{str(iou_v).replace('.', '-')}_by_class.pdf")
    plt.close()
