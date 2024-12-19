import json
import numpy as np
import matplotlib.pyplot as plt


with open("results_iou.json", "r") as f:
    data = json.load(f)

bests = np.array(data["bests_avg"])
misses = np.array(data["invalids_summed"])[:, 0] / 329_846
thresholds = np.arange(0, 1, 0.05)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.plot(thresholds, bests[:, 0])
ax1.set(ylabel="Avg. Best Score")

ax3.plot(thresholds, bests[:, 1])
ax3.set(xlabel="Min. required IoU-Overlap", ylabel="Avg. IoU-Overlap of best scored BB")

ax2.plot(bests[:, 1], bests[:, 0])

ax4.plot(bests[:, 1], misses)
ax4.set(xlabel="Avg. IoU-Overlap of best scored BB", ylabel="Ratio Images without valid BB")

plt.show()
