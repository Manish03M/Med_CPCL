# evaluation/visualize.py
# Publication-quality figures for Med-CPCL thesis.
# Generates 5 figures covering all thesis results.

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from config import FIGURES_DIR, TABLES_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    "finetuning": "#e74c3c",
    "er":         "#3498db",
    "ewc":        "#e67e22",
    "medcpcl":    "#2ecc71",
    "standard":   "#e74c3c",
    "weighted":   "#2ecc71",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

def load(fname):
    with open(os.path.join(TABLES_DIR, fname)) as f:
        return json.load(f)


# ── Figure 1: Accuracy Matrix Heatmaps (2×2) ──────────────────────────────
def fig1_accuracy_matrices():
    methods = [
        ("Fine-Tuning",       "finetuning_results.json",  COLORS["finetuning"]),
        ("Experience Replay", "replay_results.json",      COLORS["er"]),
        ("EWC",               "ewc_results.json",          COLORS["ewc"]),
        ("Med-CPCL",          "medcpcl_results.json",      COLORS["medcpcl"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Accuracy Matrix: A[i][j] = Acc on Task j after Training Task i",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (name, fname, color) in zip(axes.flat, methods):
        r   = load(fname)
        mat = np.array(r["acc_matrix"])
        sns.heatmap(mat, ax=ax, annot=True, fmt=".3f",
                    cmap="RdYlGn", vmin=0, vmax=1,
                    linewidths=0.5, linecolor="gray",
                    cbar_kws={"shrink": 0.8})
        ax.set_title(f"{name}  (AA={r['AA']:.3f}, BWT={r['BWT']:.3f})",
                     color=color, fontweight="bold")
        ax.set_xlabel("Evaluated on Task j")
        ax.set_ylabel("After training Task i")
        ax.set_xticklabels([f"T{i}" for i in range(4)])
        ax.set_yticklabels([f"T{i}" for i in range(4)], rotation=0)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_accuracy_matrices.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 2: AA / BWT / FM Bar Chart ─────────────────────────────────────
def fig2_metric_comparison():
    methods = [
        ("Fine-Tuning",       "finetuning_results.json",  COLORS["finetuning"]),
        ("ER",                "replay_results.json",      COLORS["er"]),
        ("EWC",               "ewc_results.json",          COLORS["ewc"]),
        ("Med-CPCL",          "medcpcl_results.json",      COLORS["medcpcl"]),
    ]

    names = [m[0] for m in methods]
    AA  = [load(m[1])["AA"]   for m in methods]
    BWT = [load(m[1])["BWT"]  for m in methods]
    FM  = [load(m[1])["FM"]   for m in methods]
    clr = [m[2] for m in methods]

    x   = np.arange(len(names))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))

    b1 = ax.bar(x - w, AA,  w, label="AA (higher=better)",  color=clr, alpha=0.9)
    b2 = ax.bar(x,     BWT, w, label="BWT (higher=better)", color=clr, alpha=0.6,
                hatch="//")
    b3 = ax.bar(x + w, FM,  w, label="FM (lower=better)",   color=clr, alpha=0.4,
                hatch="xx")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Metric Value")
    ax.set_title("Continual Learning Metrics Comparison", fontweight="bold")
    ax.set_ylim(-1.15, 0.85)
    ax.legend(loc="upper right", fontsize=9)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_cl_metrics.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 3: CP Coverage & Set Size (Standard vs Weighted) ───────────────
def fig3_cp_comparison():
    r     = load("medcpcl_results.json")
    final = r["cp_results"]["3"]
    tasks = [0, 1, 2, 3]

    std_cov = [final[str(t)]["standard"]["coverage"]  for t in tasks]
    wt_cov  = [final[str(t)]["weighted"]["coverage"]   for t in tasks]
    std_sz  = [final[str(t)]["standard"]["set_size"]   for t in tasks]
    wt_sz   = [final[str(t)]["weighted"]["set_size"]   for t in tasks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Standard CP vs Med-CPCL (Weighted CP) — After Task 3",
                 fontweight="bold")

    x = np.arange(4)
    w = 0.35

    # Coverage
    ax1.bar(x - w/2, std_cov, w, label="Standard CP",
            color=COLORS["standard"], alpha=0.85)
    ax1.bar(x + w/2, wt_cov,  w, label="Med-CPCL (Weighted)",
            color=COLORS["weighted"], alpha=0.85)
    ax1.axhline(0.90, color="navy", linewidth=1.5, linestyle="--",
                label="Target coverage (90%)")
    ax1.set_ylim(0.85, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Task {t}" for t in tasks])
    ax1.set_ylabel("Marginal Coverage")
    ax1.set_title("Coverage (≥0.90 required)")
    ax1.legend(fontsize=9)
    # FIX (Bug 3): use w/2 for offsets instead of hardcoded 0.175
    for i, (std_val, wt_val) in enumerate(zip(std_cov, wt_cov)):
        ax1.text(i - w/2, std_val + 0.002, f"{std_val:.3f}", ha="center",
                 fontsize=8, color="darkred")
        ax1.text(i + w/2, wt_val + 0.002, f"{wt_val:.3f}", ha="center",
                 fontsize=8, color="darkgreen")

    # Set size
    ax2.bar(x - w/2, std_sz, w, label="Standard CP",
            color=COLORS["standard"], alpha=0.85)
    ax2.bar(x + w/2, wt_sz,  w, label="Med-CPCL (Weighted)",
            color=COLORS["weighted"], alpha=0.85)
    ax2.axhline(1.0, color="gray", linewidth=1, linestyle=":",
                label="Ideal set size = 1")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Task {t}" for t in tasks])
    ax2.set_ylabel("Average Prediction Set Size")
    ax2.set_title("Set Size (lower = more efficient)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_cp_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 4: ECE Calibration Comparison ──────────────────────────────────
def fig4_ece_comparison():
    cal   = load("calibration_baselines.json")
    tasks = [0, 1, 2, 3]
    accs  = [cal["temperature"][str(t)]["accuracy"]   for t in tasks]
    raw   = [cal["temperature"][str(t)]["ece_before"] for t in tasks]
    ts    = [cal["temperature"][str(t)]["ece_after"]  for t in tasks]
    ps    = [cal["platt"][str(t)]["ece_after"]        for t in tasks]

    x = np.arange(4)
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - w,   raw, w, label="Raw (no calibration)",
           color="#e74c3c", alpha=0.85)
    ax.bar(x,       ts,  w, label="Temperature Scaling",
           color="#f39c12", alpha=0.85)
    ax.bar(x + w,   ps,  w, label="Platt Scaling",
           color="#9b59b6", alpha=0.85)

    ax2 = ax.twinx()
    ax2.plot(x, accs, "ko--", linewidth=1.5, markersize=6,
             label="Task Accuracy")
    ax2.set_ylabel("Task Accuracy", color="black")
    ax2.set_ylim(0, 1.1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Task {t}" for t in tasks])
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("Calibration Baselines: ECE vs Task Accuracy",
                 fontweight="bold")
    ax.set_ylim(0, 0.6)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    note = ("Note: Platt Scaling achieves low ECE on T1 (0.104) "
            "despite acc=0.16 -- ECE alone cannot detect drift failures.")
    ax.text(0.5, -0.13, note, transform=ax.transAxes,
            ha="center", fontsize=8, color="gray", style="italic")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_ece_calibration.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 5: EWC Fisher Degeneracy ───────────────────────────────────────
def fig5_fisher_degeneracy():
    fisher_vals = [0.000000, 0.000002, 0.000008, 0.000000]
    tasks = ["Task 0", "Task 1", "Task 2", "Task 3"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(tasks, fisher_vals, color=COLORS["ewc"], alpha=0.85,
                  edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Mean Fisher Information Magnitude")
    # FIX (Bug 1): replaced literal newline with \n escape sequence
    ax.set_title("EWC Fisher Degeneracy: Near-Zero Fisher Values\n"
                 "Cause: High-confidence convergence (train_acc≈0.999) → grad≈0",
                 fontweight="bold")
    ax.set_ylim(0, max(fisher_vals) * 3 + 1e-8)

    for bar, val in zip(bars, fisher_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e-9,
                f"{val:.6f}", ha="center", va="bottom", fontsize=9)

    # FIX (Bug 2): replaced literal newline with \n escape sequence
    ax.text(0.5, 0.75,
            "Fisher ≈ 0 → EWC penalty ≈ 0 → No protection against forgetting\n"
            "Result: EWC performance ≈ Fine-Tuning (BWT = -0.982 vs -0.983)",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_ewc_fisher_degeneracy.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("Generating thesis figures...")
    fig1_accuracy_matrices()
    fig2_metric_comparison()
    fig3_cp_comparison()
    fig4_ece_comparison()
    fig5_fisher_degeneracy()
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    import os
    for f in sorted(os.listdir(FIGURES_DIR)):
        size = os.path.getsize(os.path.join(FIGURES_DIR, f)) // 1024
        print(f"  {f}  ({size} KB)")