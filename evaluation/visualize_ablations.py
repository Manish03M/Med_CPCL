# evaluation/visualize_ablations.py
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import FIGURES_DIR, TABLES_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

with open(os.path.join(TABLES_DIR, "ablation_results.json")) as f:
    data = json.load(f)["ablations"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Med-CPCL Ablation Studies", fontweight="bold", fontsize=13)

TARGET = 0.90

# A1: Drift Compensation
ax = axes[0]
r  = data["A1_drift"]
labels   = [x["label"].replace(" (proposed)", "*") for x in r]
avg_cov  = [x["avg_cov"] for x in r]
min_cov  = [x["min_cov"] for x in r]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, avg_cov, w, label="Avg Coverage", color=["#2ecc71","#e74c3c"], alpha=0.85)
ax.bar(x + w/2, min_cov, w, label="Min Coverage", color=["#27ae60","#c0392b"], alpha=0.85)
ax.axhline(TARGET, color="navy", linewidth=1.5, linestyle="--", label="Target 90%")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel("Coverage")
ax.set_title("A1: Drift Compensation")
ax.legend(fontsize=8)
for i, (a, m) in enumerate(zip(avg_cov, min_cov)):
    ax.text(i - w/2, a + 0.002, f"{a:.3f}", ha="center", fontsize=8)
    ax.text(i + w/2, m + 0.002, f"{m:.3f}", ha="center", fontsize=8)

# A2: Gamma
ax = axes[1]
r  = data["A2_gamma"]
labels  = [x["label"].replace(" (proposed)", "*") for x in r]
min_cov = [x["min_cov"] for x in r]
avg_sz  = [x["avg_sz"]  for x in r]
x = np.arange(len(labels))
ax2b = ax.twinx()
ax.bar(x, min_cov, 0.4, label="Min Coverage", color=["#2ecc71","#e74c3c"], alpha=0.85)
ax2b.plot(x, avg_sz, "ko--", linewidth=1.5, markersize=7, label="Avg Set Size")
ax.axhline(TARGET, color="navy", linewidth=1.5, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel("Min Coverage")
ax2b.set_ylabel("Avg Set Size")
ax2b.set_ylim(5, 9)
ax.set_title("A2: Time-Aware Weighting (Gamma)")
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2b.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8)
for i, m in enumerate(min_cov):
    ax.text(i, m + 0.002, f"{m:.3f}", ha="center", fontsize=8)

# A3: Buffer Size
ax = axes[2]
r   = data["A3_buffer"]
bufsizes = [x["buf_size"] for x in r]
AA       = [x["AA"]       for x in r]
min_cov  = [x["min_cov"]  for x in r]
x = np.arange(len(bufsizes))
ax3b = ax.twinx()
bars = ax.bar(x, min_cov, 0.4,
              color=["#e74c3c","#2ecc71","#e67e22"], alpha=0.85,
              label="Min Coverage")
ax3b.plot(x, AA, "ks--", linewidth=1.5, markersize=7, label="AA (accuracy)")
ax.axhline(TARGET, color="navy", linewidth=1.5, linestyle="--", label="Target 90%")
ax.set_xticks(x)
ax.set_xticklabels([f"Buffer={b}" for b in bufsizes], fontsize=9)
ax.set_ylim(0.80, 1.02)
ax.set_ylabel("Min Coverage")
ax3b.set_ylabel("Average Accuracy (AA)")
ax3b.set_ylim(0, 1.0)
ax.set_title("A3: Replay Buffer Size")
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax3b.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8)
for i, (m, a) in enumerate(zip(min_cov, AA)):
    ax.text(i, m + 0.003, f"{m:.3f}", ha="center", fontsize=8)
    ax3b.text(i, a + 0.02, f"{a:.3f}", ha="center", fontsize=8, color="black")

plt.tight_layout()
path = os.path.join(FIGURES_DIR, "fig6_ablations.png")
plt.savefig(path)
plt.close()
print("  Saved: " + path)
