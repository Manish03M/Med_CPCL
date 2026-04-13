# evaluation/visualize_crossdataset.py
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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

def load(fname):
    with open(os.path.join(TABLES_DIR, fname)) as f:
        return json.load(f)

TARGET = 0.90

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Med-CPCL Cross-Dataset Generalisation: BloodMNIST vs OrganAMNIST",
             fontweight="bold", fontsize=13)

# ── Panel 1: Accuracy Matrix side by side ─────────────────────────────────
blood  = load("medcpcl_results.json")
organ  = load("organamnist_results.json")

ax = axes[0]
blood_mat = np.array(blood["acc_matrix"])
organ_mat = np.array(organ["acc_matrix"])

# Show both as side-by-side heatmaps in one axis using imshow
combined = np.zeros((4, 9))
combined[:, :4] = blood_mat
combined[:, 4]  = np.nan
combined[:, 5:] = organ_mat

im = ax.imshow(combined, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_title("Accuracy Matrix A[i][j] (Left: BloodMNIST | Right: OrganAMNIST)")
ax.set_yticks(range(4))
ax.set_yticklabels(["T0","T1","T2","T3"])
ax.set_xticks(range(9))
ax.set_xticklabels(["T0","T1","T2","T3","","T0","T1","T2","T3"], fontsize=9)
ax.axvline(4, color="white", linewidth=3)

for i in range(4):
    for j in range(4):
        if not np.isnan(combined[i,j]):
            ax.text(j, i, f"{combined[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
        if not np.isnan(combined[i,j+5]) and j < 4:
            ax.text(j+5, i, f"{combined[i,j+5]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
plt.colorbar(im, ax=ax, shrink=0.8)

# ── Panel 2: CL Metrics Comparison ────────────────────────────────────────
ax = axes[1]
ms    = load("multiseed_results.json")["results"]
datasets  = ["BloodMNIST", "OrganAMNIST"]
AA_vals   = [ms["Med-CPCL"]["AA"]["mean"],   organ["AA"]]
AA_stds   = [ms["Med-CPCL"]["AA"]["std"],    0.0]
BWT_vals  = [ms["Med-CPCL"]["BWT"]["mean"],  organ["BWT"]]
BWT_stds  = [ms["Med-CPCL"]["BWT"]["std"],   0.0]

x = np.arange(2)
w = 0.3
b1 = ax.bar(x - w/2, AA_vals, w, yerr=AA_stds, capsize=5,
            label="AA", color=["#3498db","#2ecc71"], alpha=0.85)
b2 = ax.bar(x + w/2, [abs(b) for b in BWT_vals], w, yerr=BWT_stds,
            capsize=5, label="|BWT| (lower=less forgetting)",
            color=["#3498db","#2ecc71"], alpha=0.5, hatch="//")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("Metric Value")
ax.set_title("CL Metrics: Med-CPCL BloodMNIST vs OrganAMNIST")
ax.legend(fontsize=9)
ax.set_ylim(0, 0.85)
for bar, val, std in zip(b1, AA_vals, AA_stds):
    label = f"{val:.3f}" + (f"+/-{std:.3f}" if std > 0 else "")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            label, ha="center", fontsize=8)

# ── Panel 3: CP Coverage Comparison ───────────────────────────────────────
ax = axes[2]

blood_final = blood["cp_results"]["3"]
organ_final = organ["cp_results"]["3"]
blood_tasks = len(blood_final)
organ_tasks = len(organ_final)

blood_wt_cov  = [blood_final[str(t)]["weighted"]["coverage"]  for t in range(blood_tasks)]
blood_std_cov = [blood_final[str(t)]["standard"]["coverage"]  for t in range(blood_tasks)]
organ_wt_cov  = [organ_final[str(t)]["weighted"]["coverage"]  for t in range(organ_tasks)]
organ_std_cov = [organ_final[str(t)]["standard"]["coverage"]  for t in range(organ_tasks)]

x_b = np.arange(blood_tasks)
x_o = np.arange(organ_tasks) + blood_tasks + 0.5
w   = 0.35

ax.bar(x_b - w/2, blood_std_cov, w, color="#e74c3c", alpha=0.7,
       label="Standard CP")
ax.bar(x_b + w/2, blood_wt_cov,  w, color="#2ecc71", alpha=0.7,
       label="Med-CPCL (Wt)")
ax.bar(x_o - w/2, organ_std_cov, w, color="#e74c3c", alpha=0.7)
ax.bar(x_o + w/2, organ_wt_cov,  w, color="#2ecc71", alpha=0.7)

ax.axhline(TARGET, color="navy", linewidth=1.5, linestyle="--",
           label="Target 90%")
ax.axvline(blood_tasks, color="gray", linewidth=1.5, linestyle=":",
           alpha=0.7)

ax.set_xticks(list(x_b) + list(x_o))
blood_lbls = [f"B-T{t}" for t in range(blood_tasks)]
organ_lbls = [f"O-T{t}" for t in range(organ_tasks)]
ax.set_xticklabels(blood_lbls + organ_lbls, fontsize=9)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel("Marginal Coverage")
ax.set_title("Coverage Guarantee Across Datasets (B=Blood, O=Organ)")
ax.legend(fontsize=8, loc="lower right")

ax.text(1.5,  0.862, "BloodMNIST",  ha="center", fontsize=9,
        color="#3498db", fontweight="bold")
ax.text(5.5,  0.862, "OrganAMNIST", ha="center", fontsize=9,
        color="#2ecc71", fontweight="bold")

plt.tight_layout()
path = os.path.join(FIGURES_DIR, "fig7_cross_dataset.png")
plt.savefig(path)
plt.close()
print("  Saved: " + path)

if __name__ == "__main__":
    pass
