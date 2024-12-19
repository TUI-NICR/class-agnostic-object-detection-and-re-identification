"""
Create some stats and graphics of co3d dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(
    "co3d_reid_v1.csv", sep=",",
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

df["xsize"] = df["xmax"] - df["xmin"]
df["ysize"] = df["ymax"] - df["ymin"]

num_imgs = len(df)

df_by_id = df.groupby("id")
num_ids = len(df_by_id)
num_img_per_id = df_by_id["img_num"].count().mean()

df_by_cls = df.groupby("class")
id_per_cls = df_by_cls["id"].nunique().sort_values()
num_img_per_cls = df_by_cls["img_num"].count().sort_values()
size_per_cls = df_by_cls[["xsize", "ysize"]].mean()
size_mean = size_per_cls.mean()
size_std = size_per_cls.std()

print(f"{num_imgs=}, {num_ids=}, {num_img_per_id=}")
print("\nsize_mean")
print(size_mean)
print("\nsize_std")
print(size_std)
fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
id_per_cls.plot(kind="barh", title="id_per_cls", fontsize=6, ax=axes[0])
num_img_per_cls.plot(kind="barh", title="num_img_per_cls", fontsize=6, ax=axes[1])
size_per_cls.plot.scatter(x="xsize", y="ysize", xlim=(0, 1000), ylim=(0, 1000), title="size_per_cls", fontsize=6, ax=axes[2])
for k, v in size_per_cls.iterrows():
    axes[2].annotate(k, v, xytext=(5, -2), textcoords='offset points', fontsize=4)
axes[2].plot(np.arange(1000), np.arange(1000))
fig.tight_layout()
fig.savefig("co3d_reid_v1.pdf")

plt.close()
fig, ax = plt.subplots()
size_per_cls.plot.scatter(x="xsize", y="ysize", xlim=(0, 1000), ylim=(0, 1000), xlabel="Breite", ylabel="Höhe", title="Abmessungen", fontsize=6, ax=ax)
for k, v in size_per_cls.iterrows():
    ax.annotate(k, v, xytext=(5, -2), textcoords='offset points', fontsize=4)
ax.plot(np.arange(1000), np.arange(1000))
fig.savefig("size.svg")
