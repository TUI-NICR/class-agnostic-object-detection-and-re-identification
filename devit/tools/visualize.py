import numpy as np
import json
import os
import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict

P = "/path/to/results"


results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
for shot_split in os.listdir(P):
    shot, split, *sw = shot_split.split("_")
    shot = int(shot.split("-")[-1])
    split = int(split.split("-")[-1])
    sw = len(sw) > 0
    p = os.path.join(P, shot_split)
    for model in os.listdir(p):
        p = os.path.join(P, shot_split, model)
        with open(os.path.join(p, "metrics.json"), "r") as f:
            metrics = json.load(f)
        inf_time = metrics["inf_time"]
        p_time = metrics["prototype_time"]
        for m, v in metrics["bbox"].items():
            if not ("AP-" in m or "AR-" in m) or "UNDEF" in m:
                continue
            metric, class_ = m.split("-")
            results[sw][shot][model][class_][split][metric] = v
            results[sw][shot][model][class_][split]["inf_time"] = inf_time
            results[sw][shot][model][class_][split]["prototype_time"] = p_time

df = pd.DataFrame.from_dict({
    (i, j, k, m, n): results[i][j][k][m][n]
        for i in results.keys()
            for j in results[i].keys()
                for k in results[i][j].keys()
                    for m in results[i][j][k].keys()
                        for n in results[i][j][k][m].keys()
    },
    orient="index"
)
df = df.rename_axis(["sliding-window", "shot", "model", "class", "split"])

paper_res = []
labels = 0
fig, axs = plt.subplots(13, 2, figsize=(10, 40))
for j, (sw, df_sw) in enumerate(df.groupby("sliding-window")):
    for (shot, df_sh), c in zip(df_sw.groupby("shot"), [["red", "orange", "green", "blue", "purple"], ["teal", "cyan", "magenta"]][j]):
        for ax in axs[:, 1]:
            ax.scatter(x=-100, y=0, c=c, marker="o", label=f"shot-{shot}{', sw' if sw else ''}")
        for (model, df_md), m in zip(df_sh.groupby("model"), ["o", "^", "s"]):
            if shot == 20 and labels < 3:
                for ax in axs[:, 1]:
                    ax.scatter(x=-100, y=0, c="black", marker=m, label=model)
                labels += 1
            for i, (class_, df_cl) in enumerate(df_md.groupby("class")):
                vs = df_cl.mean()
                axs[i, 0].scatter(x=vs["inf_time"] + vs["prototype_time"], y=vs["AP"], c=c, marker=m)
                axs[i, 0].set_ylabel("AP")
                axs[i, 0].set_xlabel("Time")
                axs[i, 0].set_title(class_)
                axs[i, 0].set_xscale("log")
                axs[i, 0].set_xlim(left=10, right=1500)
                axs[i, 1].scatter(x=vs["inf_time"] + vs["prototype_time"], y=vs["AR"], c=c, marker=m)
                axs[i, 1].set_ylabel("AR")
                axs[i, 1].set_xlabel("Time")
                axs[i, 1].set_title(class_)
                axs[i, 1].set_xscale("log")
                axs[i, 1].set_xlim(left=10, right=1500)
            vs = df_md.mean()
            if not sw:
                paper_res.append(dict(n=shot, model=model, x=float(vs["inf_time"]), y=float(vs["AP"])))
            axs[12, 0].scatter(x=vs["inf_time"] + vs["prototype_time"], y=vs["AP"], c=c, marker=m)
            axs[12, 0].set_ylabel("AP")
            axs[12, 0].set_xlabel("Time")
            axs[12, 0].set_title("Avg")
            axs[12, 0].set_xscale("log")
            axs[12, 0].set_xlim(left=10, right=1500)
            axs[12, 1].scatter(x=vs["inf_time"] + vs["prototype_time"], y=vs["AR"], c=c, marker=m)
            axs[12, 1].set_ylabel("AR")
            axs[12, 1].set_xlabel("Time")
            axs[12, 1].set_title("Avg")
            axs[12, 1].set_xscale("log")
            axs[12, 1].set_xlim(left=10, right=1500)
for ax in axs[:, 1]:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.tight_layout()
fig.savefig("test.pdf")

with open("paper_res.json", "w") as f:
    json.dump(paper_res, f)
