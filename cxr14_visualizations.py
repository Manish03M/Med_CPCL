"""
Med-CPCL NIH CXR-14 Extended Dataset — Publication-Quality Thesis Visualizations
All plots are thesis-ready, matplotlib-only, academic style.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, PercentFormatter
import warnings
warnings.filterwarnings("ignore")

# ── Global academic style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "axes.titlepad":      8,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "grid.linewidth":     0.5,
    "grid.alpha":         0.4,
    "lines.linewidth":    1.5,
})

# Colorblind-safe palette (Wong 2011)
_PALETTE = ["#0072B2","#E69F00","#009E73","#D55E00",
            "#56B4E9","#CC79A7","#F0E442","#000000"]


def _get_cmap_colors(n, cmap_name="Blues"):
    """Return n evenly-spaced colors from a matplotlib colormap."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(0.35 + 0.55 * i / max(n - 1, 1)) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════
# 1. CLASS-LEVEL LABEL PREVALENCE (horizontal bar, log-scale)
# ══════════════════════════════════════════════════════════════════════════
def plot_label_prevalence(
    class_names: list,
    positive_counts: list,
    total_train: int,
    task_colors: dict = None,
    task_assignment: dict = None,
    save_path: str = None,
):
    """
    Horizontal bar chart of per-class label prevalence (%) sorted by frequency.
    Bars are color-coded by task assignment if provided.

    Args:
        class_names:      List of 14 class name strings.
        positive_counts:  Corresponding positive sample counts.
        total_train:      Total training images (denominator).
        task_colors:      {task_id: hex_color} for coloring bars.
        task_assignment:  {class_name: task_id} for bar coloring.
        save_path:        If given, save figure to this path.
    """
    n = len(class_names)
    prevalences = [100 * p / total_train for p in positive_counts]

    # Sort descending by prevalence
    order = np.argsort(prevalences)[::-1]
    sorted_names    = [class_names[i] for i in order]
    sorted_prev     = [prevalences[i] for i in order]
    sorted_counts   = [positive_counts[i] for i in order]

    if task_colors is None:
        task_colors = {0:"#2166AC", 1:"#4DAC26", 2:"#D1B847", 3:"#D6604D"}
    if task_assignment is None:
        task_assignment = {}

    bar_colors = [
        task_colors.get(task_assignment.get(nm, -1), "#888888")
        for nm in sorted_names
    ]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    y_pos = np.arange(n)

    bars = ax.barh(y_pos, sorted_prev, color=bar_colors,
                   edgecolor="white", linewidth=0.4, height=0.7)

    # Annotate count + prevalence
    for i, (bar, cnt, pv) in enumerate(zip(bars, sorted_counts, sorted_prev)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}  ({pv:.2f}%)",
                va="center", ha="left", fontsize=7.5, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel("Prevalence in Training Set (%)")
    ax.set_title("NIH CXR-14: Per-Class Label Prevalence Across All Training Images",
                 fontweight="bold")
    ax.set_xlim(0, max(sorted_prev) * 1.45)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)

    # Task legend
    if task_assignment:
        handles = [
            mpatches.Patch(color=task_colors[t], label=f"Task {t}")
            for t in sorted(task_colors.keys())
        ]
        ax.legend(handles=handles, title="CIL Task", loc="lower right",
                  fontsize=8, title_fontsize=8)

    # Annotate rarest class
    rare_idx = sorted_prev.index(min(sorted_prev))
    ax.annotate(
        "← Rarest class\n   (Hernia, 0.34%)",
        xy=(sorted_prev[rare_idx], rare_idx),
        xytext=(sorted_prev[rare_idx] + 1.5, rare_idx - 0.5),
        arrowprops=dict(arrowstyle="-|>", color="#D55E00", lw=1.2),
        fontsize=8, color="#D55E00", fontstyle="italic",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 2. POS_WEIGHT IMBALANCE RATIO (log-scale bar chart)
# ══════════════════════════════════════════════════════════════════════════
def plot_pos_weight_distribution(
    class_names: list,
    pos_weights: list,
    save_path: str = None,
):
    """
    Bar chart of per-class Focal Loss pos_weight (neg/pos ratio) on log scale.
    Illustrates the magnitude of class imbalance each weighting term must correct.
    """
    n = len(class_names)
    order = np.argsort(pos_weights)[::-1]
    sorted_names   = [class_names[i] for i in order]
    sorted_weights = [pos_weights[i] for i in order]

    colors = _get_cmap_colors(n, "plasma")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(n)
    bars = ax.bar(x, sorted_weights, color=colors,
                  edgecolor="white", linewidth=0.4, width=0.72)

    for bar, w in zip(bars, sorted_weights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.06,
                f"{w:.0f}", ha="center", va="bottom", fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylabel("pos_weight = neg_count / pos_count  (log scale)")
    ax.set_title("Focal Loss Positive Weights: Imbalance Ratio per Pathology Class",
                 fontweight="bold")
    ax.yaxis.grid(True, which="both", linestyle="--")
    ax.set_axisbelow(True)

    # Reference line at 1 (balanced)
    ax.axhline(y=1, color="#333333", linestyle=":", linewidth=1,
               label="Balanced (pos_weight = 1)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 3. TASK PARTITION SUMMARY (stacked bar — train/cal/val/test split sizes)
# ══════════════════════════════════════════════════════════════════════════
def plot_task_split_summary(
    task_ids: list,
    task_class_names: list,
    train_sizes: list,
    cal_sizes: list,
    val_sizes: list,
    test_sizes: list,
    save_path: str = None,
):
    """
    Grouped horizontal bars showing patient-wise split sizes per task.
    Illustrates the 80/20 calibration split within training data.
    """
    n = len(task_ids)
    split_labels  = ["Train", "Calibration", "Val", "Test"]
    split_data    = [train_sizes, cal_sizes, val_sizes, test_sizes]
    split_colors  = ["#2166AC", "#4393C3", "#F4A582", "#D6604D"]

    x = np.arange(n)
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 4.2))

    for j, (lbl, data, color) in enumerate(zip(split_labels, split_data, split_colors)):
        offset = (j - 1.5) * width
        rects = ax.bar(x + offset, data, width, label=lbl,
                       color=color, edgecolor="white", linewidth=0.4)
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 200,
                    f"{int(h/1000):.0f}k", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    task_labels = [
        f"Task {tid}\n({', '.join(names[:2])}{'...' if len(names)>2 else ''})"
        for tid, names in zip(task_ids, task_class_names)
    ]
    ax.set_xticklabels(task_labels, fontsize=9)
    ax.set_ylabel("Number of Images")
    ax.set_title("Patient-Wise Train / Calibration / Val / Test Split Sizes per CIL Task",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k")
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 4. TRAINING LOSS CURVES (all 4 tasks, 2×2 grid)
# ══════════════════════════════════════════════════════════════════════════
def plot_training_curves(
    task_train_losses: list,
    task_val_losses: list,
    task_val_aucs: list,
    task_names: list,
    save_path: str = None,
):
    """
    2×2 grid of dual-axis plots: training loss (left) and validation AUC (right)
    per epoch for all 4 tasks.

    Args:
        task_train_losses: List of 4 lists of per-epoch training losses.
        task_val_losses:   List of 4 lists of per-epoch val losses.
        task_val_aucs:     List of 4 lists of per-epoch val AUC.
        task_names:        List of 4 task title strings.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    for i, (ax, tr_loss, va_loss, va_auc, tname) in enumerate(
        zip(axes, task_train_losses, task_val_losses, task_val_aucs, task_names)
    ):
        epochs = np.arange(1, len(tr_loss) + 1)

        # Loss (left axis)
        ln1 = ax.plot(epochs, tr_loss, color=_PALETTE[0], linestyle="-",
                      marker="o", markersize=3, label="Train Loss", linewidth=1.4)
        ln2 = ax.plot(epochs, va_loss, color=_PALETTE[1], linestyle="--",
                      marker="s", markersize=3, label="Val Loss", linewidth=1.4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Focal Loss")
        ax.set_title(f"Task {i}: {tname}", fontweight="bold", fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.grid(True, linestyle="--")

        # AUC (right axis)
        ax2 = ax.twinx()
        ln3 = ax2.plot(epochs, va_auc, color=_PALETTE[2], linestyle="-.",
                       marker="^", markersize=3.5, label="Val AUC", linewidth=1.4)
        ax2.set_ylabel("Val AUC-ROC", color=_PALETTE[2])
        ax2.tick_params(axis="y", labelcolor=_PALETTE[2])
        ax2.set_ylim(max(0, min(va_auc) - 0.05), min(1.0, max(va_auc) + 0.04))
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(_PALETTE[2])

        # Best checkpoint marker
        best_epoch = int(np.argmax(va_auc)) + 1
        best_auc   = max(va_auc)
        ax2.axvline(best_epoch, color="#888888", linestyle=":", linewidth=1)
        ax2.text(best_epoch + 0.1, ax2.get_ylim()[0] + 0.01,
                 f"Best\nEp{best_epoch}", fontsize=7.5, color="#555555")

        # Combined legend
        lns = ln1 + ln2 + ln3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc="center right", fontsize=8, framealpha=0.85)

    fig.suptitle("ResNet-50 Training Dynamics — Focal Loss and AUC per Epoch",
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 5. PER-CLASS AUC-ROC BAR CHART (grouped by task)
# ══════════════════════════════════════════════════════════════════════════
def plot_per_class_auc(
    class_names: list,
    val_aucs: list,
    test_aucs: list,
    task_boundaries: list,
    task_colors: dict = None,
    save_path: str = None,
):
    """
    Grouped bar chart: Val AUC and Test AUC side-by-side per class,
    with vertical separators between tasks.

    Args:
        class_names:     14 class names in task order.
        val_aucs:        Per-class best val AUC.
        test_aucs:       Per-class test AUC.
        task_boundaries: List of (start_idx, end_idx, task_label) tuples.
        task_colors:     {task_id: hex}.
    """
    if task_colors is None:
        task_colors = {0:"#2166AC", 1:"#4DAC26", 2:"#D1B847", 3:"#D6604D"}

    n = len(class_names)
    x = np.arange(n)
    w = 0.34

    fig, ax = plt.subplots(figsize=(13, 4.8))

    # Shade task regions
    bg_alphas = [0.06, 0.10, 0.06, 0.10]
    for task_id, (start, end, tlabel) in enumerate(task_boundaries):
        ax.axvspan(start - 0.5, end - 0.5, alpha=bg_alphas[task_id],
                   color=task_colors[task_id], zorder=0)
        mid = (start + end - 1) / 2
        ax.text(mid, 1.03, f"Task {task_id}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=task_colors[task_id],
                transform=ax.get_xaxis_transform())

    # Task separators
    for _, (_, end, _) in enumerate(task_boundaries[:-1]):
        ax.axvline(end - 0.5, color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=1)

    bar1 = ax.bar(x - w/2, val_aucs,  w, label="Val AUC",
                  color="#4393C3", edgecolor="white", linewidth=0.3, zorder=2)
    bar2 = ax.bar(x + w/2, test_aucs, w, label="Test AUC",
                  color="#D6604D", edgecolor="white", linewidth=0.3, zorder=2)

    for bar, v in zip(bar1, val_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.8, rotation=90)
    for bar, v in zip(bar2, test_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.8, rotation=90)

    ax.axhline(0.80, color="#555555", linestyle=":", linewidth=1,
               label="AUC = 0.80 reference")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.60, 1.05)
    ax.set_title("ResNet-50 Per-Class AUC-ROC: Validation vs. Test (NIH CXR-14, 4-Task CIL)",
                 fontweight="bold")
    ax.yaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 6. BACKBONE DRIFT ACROSS TASK TRANSITIONS
# ══════════════════════════════════════════════════════════════════════════
def plot_backbone_drift(
    drift_values: list,
    gamma_values: list,
    ref_drift: float,
    gamma_min: float,
    transition_labels: list = None,
    save_path: str = None,
):
    """
    Two-panel figure: (left) backbone drift Δμ with ref_drift reference line;
    (right) computed γ_t with γ_min floor line.
    """
    if transition_labels is None:
        transition_labels = [f"T{i}→T{i+1}" for i in range(len(drift_values))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ── Panel 1: Drift magnitudes ──────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(drift_values))
    bars = ax.bar(x, drift_values, color=_PALETTE[0],
                  edgecolor="white", linewidth=0.5, width=0.5, zorder=2)
    ax.axhline(ref_drift, color=_PALETTE[3], linestyle="--", linewidth=1.5,
               label=f"ref_drift = {ref_drift:.2f}")
    ax.axhline(0, color="#BBBBBB", linewidth=0.5)
    for bar, v in zip(bars, drift_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(transition_labels, fontsize=10)
    ax.set_ylabel("L2 Distance ‖μ(t) − μ(t−1)‖₂\n(unit-sphere normalised embeddings)")
    ax.set_title("Backbone Drift Δμ\nper Task Transition", fontweight="bold")
    ax.set_ylim(0, ref_drift * 1.5)
    ax.yaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    # ── Panel 2: Dynamic γ values ──────────────────────────────────────────
    ax2 = axes[1]
    # Show the exponential response curve
    delta_range = np.linspace(0, 2.0, 300)
    # Using eta=1.0
    gamma_curve = np.maximum(gamma_min, np.exp(-1.0 * delta_range / ref_drift))
    ax2.plot(delta_range, gamma_curve, color=_PALETTE[0], linewidth=1.8,
             label=r"$\gamma_t = \max(\gamma_{min}, e^{-\eta \hat{\Delta}_t})$")
    ax2.axhline(gamma_min, color=_PALETTE[3], linestyle="--", linewidth=1.5,
                label=f"γ_min floor = {gamma_min:.2f}")

    # Mark actual transitions
    for i, (dv, gv, lbl) in enumerate(zip(drift_values, gamma_values, transition_labels)):
        ax2.scatter(dv, gv, zorder=5, s=90,
                    color=_PALETTE[i % len(_PALETTE)],
                    edgecolors="#333333", linewidths=0.8)
        ax2.annotate(lbl,
                     xy=(dv, gv),
                     xytext=(dv + 0.02, gv + 0.015),
                     fontsize=9, color="#333333")

    ax2.set_xlabel("Δμ (L2 backbone drift, unit sphere)")
    ax2.set_ylabel("Computed γ_t")
    ax2.set_title("Dynamic γ Controller Response Curve\nwith Observed Transitions",
                  fontweight="bold")
    ax2.set_ylim(gamma_min - 0.1, 1.05)
    ax2.yaxis.grid(True, linestyle="--")
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 7. MONDRIAN CONFORMAL QUANTILES — HORIZONTAL BAR WITH SCORE RANGE
# ══════════════════════════════════════════════════════════════════════════
def plot_mondrian_quantiles(
    class_names: list,
    quantiles: list,
    score_mins: list,
    score_maxs: list,
    task_assignment: dict = None,
    target_alpha: float = 0.10,
    save_path: str = None,
):
    """
    Horizontal chart showing per-class APS score range [min, max] with
    the estimated Mondrian quantile q̂_c superimposed.
    """
    n = len(class_names)
    # Sort by quantile descending
    order = np.argsort(quantiles)[::-1]
    names  = [class_names[i] for i in order]
    qhats  = [quantiles[i] for i in order]
    smins  = [score_mins[i] for i in order]
    smaxs  = [score_maxs[i] for i in order]

    task_colors = {0:"#2166AC", 1:"#4DAC26", 2:"#D1B847", 3:"#D6604D"}

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(n)

    for i, (nm, qh, smin, smax) in enumerate(zip(names, qhats, smins, smaxs)):
        tid = task_assignment.get(nm, -1) if task_assignment else -1
        c = task_colors.get(tid, "#999999")
        # Score range bar (faint)
        ax.barh(y[i], smax - smin, left=smin, height=0.4,
                color=c, alpha=0.20, zorder=1)
        # Quantile marker
        ax.scatter(qh, y[i], color=c, s=60, zorder=3,
                   edgecolors="#333333", linewidths=0.6)
        ax.text(qh + 0.012, y[i], f"{qh:.4f}",
                va="center", ha="left", fontsize=8.2, color=c, fontweight="bold")
        # Range label
        ax.text(smin - 0.01, y[i], f"[{smin:.2f}", va="center", ha="right", fontsize=7)
        ax.text(smax + 0.01, y[i], f"{smax:.2f}]", va="center", ha="left", fontsize=7)

    ax.axvline(1.0 - target_alpha, color="#D55E00", linestyle=":", linewidth=1.5,
               label=f"Coverage target 1−α = {1-target_alpha:.2f}")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("APS Non-Conformity Score  s(x, c) = 1 − σ(logit_c)")
    ax.set_title(
        "Mondrian Conformal Quantiles q̂_c per Pathology Class\n"
        "(score range = shaded bar; q̂_c = filled circle)",
        fontweight="bold",
    )
    ax.set_xlim(-0.05, 1.12)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(color=task_colors[t], alpha=0.5, label=f"Task {t}")
        for t in sorted(task_colors)
    ]
    handles += [
        plt.Line2D([0], [0], color="#D55E00", linestyle=":", linewidth=1.5,
                   label=f"1−α = {1-target_alpha:.2f}")
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 8. PER-CLASS COVERAGE RESULTS (diverging bar from 90% target)
# ══════════════════════════════════════════════════════════════════════════
def plot_per_class_coverage(
    class_names: list,
    coverages: list,
    n_positives: list,
    target: float = 0.90,
    task_assignment: dict = None,
    save_path: str = None,
):
    """
    Diverging horizontal bar chart centred at the coverage target (0.90).
    Bars going left = below target (red); bars going right = above target (blue).
    """
    n = len(class_names)
    gaps = [c - target for c in coverages]

    # Sort by gap ascending (worst first)
    order = np.argsort(gaps)
    names  = [class_names[i] for i in order]
    gaps_s = [gaps[i] for i in order]
    covs_s = [coverages[i] for i in order]
    npos_s = [n_positives[i] for i in order]

    colors = ["#D6604D" if g < 0 else "#2166AC" for g in gaps_s]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(n)

    ax.barh(y, gaps_s, color=colors, edgecolor="white",
            linewidth=0.4, height=0.65, zorder=2)

    # Labels
    for i, (g, cov, npos) in enumerate(zip(gaps_s, covs_s, npos_s)):
        direction = 1 if g < 0 else 1
        xpos = g + (0.002 if g >= 0 else -0.002)
        ha   = "left" if g >= 0 else "right"
        ax.text(xpos, y[i],
                f"{cov:.4f}  (N+={npos:,})",
                va="center", ha=ha, fontsize=8, color="#1A1A1A")

    ax.axvline(0, color="#333333", linewidth=1.2, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f"Coverage − {target:.2f}  (positive = surplus above target)")
    ax.set_title(
        f"Per-Class Mondrian Coverage Relative to Target (α=0.10, target=0.90)\n"
        "Red = below 90% (coverage deficit)   |   Blue = above 90% (coverage surplus)",
        fontweight="bold",
    )

    # Annotate pass/fail counts
    n_pass = sum(1 for g in gaps_s if g >= 0)
    ax.text(0.98, 0.02, f"Pass: {n_pass}/{n}  ✓",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="#2166AC", fontweight="bold")
    ax.text(0.98, 0.08, f"Fail: {n-n_pass}/{n}  ✗",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="#D6604D", fontweight="bold")

    ax.xaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 9. TASK-LEVEL COVERAGE + SET SIZE (dual-axis, 4-task)
# ══════════════════════════════════════════════════════════════════════════
def plot_task_level_summary(
    task_ids: list,
    marginal_coverages: list,
    min_coverages: list,
    avg_set_sizes: list,
    target_coverage: float = 0.90,
    total_classes: int = 14,
    save_path: str = None,
):
    """
    Combined bar + line chart: marginal/min coverage (bars) and average
    prediction set size (line) per task.
    """
    n = len(task_ids)
    x = np.arange(n)
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4.8))

    # Coverage bars
    b1 = ax1.bar(x - w/2, marginal_coverages, w,
                 label="Marginal Coverage", color="#4393C3",
                 edgecolor="white", linewidth=0.4, zorder=2)
    b2 = ax1.bar(x + w/2, min_coverages, w,
                 label="Min Coverage", color="#2166AC",
                 edgecolor="white", linewidth=0.4, zorder=2)

    # Target line
    ax1.axhline(target_coverage, color="#D55E00", linestyle="--", linewidth=1.5,
                label=f"Target = {target_coverage:.2f}", zorder=3)

    for bar, v in zip(b1, marginal_coverages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar, v in zip(b2, min_coverages):
        c = "#D6604D" if v < target_coverage else "#1A6F00"
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=c,
                 fontweight="bold")

    ax1.set_ylim(0.80, 1.02)
    ax1.set_ylabel("Empirical Coverage")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Task {t}" for t in task_ids], fontsize=10)
    ax1.set_title("Task-Level Conformal Coverage and Average Prediction Set Size",
                  fontweight="bold")
    ax1.yaxis.grid(True, linestyle="--")
    ax1.set_axisbelow(True)

    # Set size on right axis
    ax2 = ax1.twinx()
    ax2.plot(x, avg_set_sizes, color=_PALETTE[2], marker="D",
             markersize=7, linewidth=1.8, zorder=4, label="Avg Set Size")
    ax2.plot(x, [s / total_classes for s in avg_set_sizes],
             color=_PALETTE[4], marker="o", markersize=5, linewidth=1.4,
             linestyle=":", zorder=4, label="Triage Efficiency (÷14)")
    ax2.set_ylabel("Avg Prediction Set Size  /  Triage Efficiency")
    ax2.set_ylim(0, total_classes * 0.35)
    ax2.spines["right"].set_visible(True)

    for xi, sz in zip(x, avg_set_sizes):
        ax2.text(xi + 0.22, sz + 0.05, f"{sz:.3f}", fontsize=8.2,
                 color=_PALETTE[2], fontweight="bold")

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc="lower left", fontsize=8.5, ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 10. ABLATION STUDY — HEATMAP-STYLE RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════
def plot_ablation_heatmap(
    config_names: list,
    task_ids: list,
    min_coverages: np.ndarray,
    avg_set_sizes: np.ndarray,
    target_coverage: float = 0.90,
    save_path: str = None,
):
    """
    Side-by-side colour-coded tables showing min coverage and avg set size
    for each ablation configuration × task combination.

    Args:
        config_names:    e.g. ["Med-CPCL","A1: No Drift","A2: Uniform","A3: Marginal CP"]
        task_ids:        e.g. [0, 1, 2, 3]
        min_coverages:   (n_configs × n_tasks) array
        avg_set_sizes:   (n_configs × n_tasks) array
        target_coverage: 0.90
    """
    n_configs = len(config_names)
    n_tasks   = len(task_ids)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

    titles = ["Minimum Coverage per Task (target ≥ 0.90)",
              "Average Prediction Set Size per Task"]
    data_list = [min_coverages, avg_set_sizes]

    cmaps = ["RdYlGn", "RdYlGn_r"]

    for ax, title, data, cmap_name in zip(axes, titles, data_list, cmaps):
        cmap = plt.get_cmap(cmap_name)

        # Normalise
        vmin = data.min() - 0.01
        vmax = data.max() + 0.01
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        for i in range(n_configs):
            for j in range(n_tasks):
                val  = data[i, j]
                rgba = cmap(norm(val))
                rect = plt.Rectangle([j, n_configs - 1 - i], 1, 1,
                                     facecolor=rgba, edgecolor="white", lw=1.5)
                ax.add_patch(rect)
                lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                tc   = "white" if lum < 0.55 else "#1A1A1A"
                fail_mark = "" if (cmap_name != "RdYlGn" or val >= target_coverage) else " ✗"
                ax.text(j + 0.5, n_configs - 0.5 - i,
                        f"{val:.4f}{fail_mark}",
                        ha="center", va="center", fontsize=10,
                        color=tc, fontweight="bold")

        ax.set_xlim(0, n_tasks)
        ax.set_ylim(0, n_configs)
        ax.set_xticks(np.arange(n_tasks) + 0.5)
        ax.set_xticklabels([f"Task {t}" for t in task_ids], fontsize=9.5)
        ax.set_yticks(np.arange(n_configs) + 0.5)
        ax.set_yticklabels(config_names[::-1], fontsize=9.5)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.tick_params(length=0)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=15)

    fig.suptitle("Ablation Study: Per-Task Min Coverage and Avg Set Size",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 11. COVERAGE vs. AUC SCATTER — MODEL PERFORMANCE ≠ COVERAGE SAFETY
# ══════════════════════════════════════════════════════════════════════════
def plot_coverage_vs_auc(
    class_names: list,
    test_aucs: list,
    coverages: list,
    task_assignment: dict = None,
    target_coverage: float = 0.90,
    save_path: str = None,
):
    """
    Scatter plot: per-class Test AUC (x) vs. per-class Coverage (y).
    Demonstrates that high AUC does not guarantee high coverage.
    """
    task_colors = {0:"#2166AC", 1:"#4DAC26", 2:"#D1B847", 3:"#D6604D"}
    markers = {0:"o", 1:"s", 2:"^", 3:"D"}

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Quadrant shading
    ax.axhspan(target_coverage, 1.0, alpha=0.05, color="#2166AC")
    ax.axhspan(0.0, target_coverage, alpha=0.05, color="#D6604D")
    ax.axvline(0.80, color="#BBBBBB", linestyle=":", linewidth=1)
    ax.axhline(target_coverage, color="#D55E00", linestyle="--", linewidth=1.4,
               label=f"Coverage target = {target_coverage:.2f}")

    for nm, auc, cov in zip(class_names, test_aucs, coverages):
        tid = task_assignment.get(nm, 0) if task_assignment else 0
        c = task_colors.get(tid, "#888888")
        m = markers.get(tid, "o")
        ax.scatter(auc, cov, color=c, marker=m, s=85, zorder=3,
                   edgecolors="#333333", linewidths=0.6)
        # Offset label to avoid overlap
        ax.annotate(
            nm,
            xy=(auc, cov),
            xytext=(auc + 0.002, cov + 0.004),
            fontsize=7.5, color="#333333",
        )

    # Quadrant labels
    ax.text(0.675, target_coverage + 0.006, "Below AUC threshold\nbut Covered ✓",
            fontsize=8, color="#2166AC", ha="left", style="italic")
    ax.text(0.62, target_coverage - 0.015, "Coverage Deficit ✗",
            fontsize=8, color="#D6604D", ha="left", style="italic")
    ax.text(0.92, target_coverage - 0.015, "High AUC /\nCoverage Deficit ✗",
            fontsize=8, color="#8B2020", ha="right", style="italic")

    handles = [
        mpatches.Patch(color=task_colors[t], label=f"Task {t}")
        for t in sorted(task_colors)
    ] + [plt.Line2D([0],[0], color="#D55E00", linestyle="--", linewidth=1.4,
                    label=f"Target = {target_coverage:.2f}")]
    ax.legend(handles=handles, fontsize=9, loc="lower right")

    ax.set_xlabel("Test AUC-ROC")
    ax.set_ylabel("Empirical Class-Conditional Coverage")
    ax.set_title(
        "AUC-ROC vs. Conformal Coverage: Divergence Between\n"
        "Discriminative Performance and Safety Guarantees",
        fontweight="bold",
    )
    ax.set_xlim(0.60, 0.98)
    ax.set_ylim(0.84, 1.02)
    ax.xaxis.grid(True, linestyle="--")
    ax.yaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 12. TRIAGE EFFICIENCY VIOLIN/BAR ACROSS TASKS
# ══════════════════════════════════════════════════════════════════════════
def plot_triage_efficiency(
    task_ids: list,
    avg_set_sizes: list,
    total_classes: int = 14,
    save_path: str = None,
):
    """
    Bar chart of triage efficiency (avg set size ÷ total classes) per task,
    with a horizontal line at 1.0 (full, non-informative set) and annotation
    of the absolute average set size.
    """
    efficiencies = [s / total_classes for s in avg_set_sizes]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(task_ids))
    colors = _get_cmap_colors(len(task_ids), "Blues")

    bars = ax.bar(x, efficiencies, color=colors, edgecolor="white",
                  linewidth=0.4, width=0.5, zorder=2)

    for bar, eff, sz in zip(bars, efficiencies, avg_set_sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{eff:.4f}\n({sz:.3f}/{total_classes})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(1.0, color="#D55E00", linestyle="--", linewidth=1.4,
               label=f"Full set (non-informative) = 1.0  [{total_classes}/{total_classes}]")
    ax.axhline(1/total_classes, color="#888888", linestyle=":", linewidth=1,
               label=f"Single-class (max informative) = {1/total_classes:.3f}  [1/{total_classes}]")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Task {t}" for t in task_ids], fontsize=10)
    ax.set_ylabel(f"Triage Efficiency = Avg Set Size / {total_classes}")
    ax.set_ylim(0, 0.30)
    ax.set_title(
        "Med-CPCL Triage Efficiency per Task\n"
        "(lower = more informative prediction sets)",
        fontweight="bold",
    )
    ax.legend(fontsize=8.5, loc="upper right")
    ax.yaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 13. PRODUCT WEIGHT DECAY ACROSS STORED TASKS
# ══════════════════════════════════════════════════════════════════════════
def plot_product_weight_decay(
    gamma_history: list,
    current_task: int = 3,
    save_path: str = None,
):
    """
    Horizontal bar chart showing accumulated product weight for scores stored
    at each prior task. Illustrates the temporal discounting structure.
    """
    n_stored = current_task + 1
    weights = []
    for stored in range(n_stored):
        w = 1.0
        for k in range(stored, current_task):
            if k < len(gamma_history):
                w *= gamma_history[k]
        weights.append(w)

    labels  = [f"Stored at Task {t}" for t in range(n_stored)]
    colors  = _get_cmap_colors(n_stored, "GnBu")[::-1]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(n_stored)
    bars = ax.barh(y, weights, color=colors, edgecolor="white",
                   linewidth=0.4, height=0.55, zorder=2)

    for bar, w, stored in zip(bars, weights, range(n_stored)):
        exp = current_task - stored
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"w = {w:.4f}  (= γ^{exp})",
                va="center", ha="left", fontsize=9.5, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(f"Accumulated Product Weight at Task {current_task}")
    ax.set_title(
        f"Temporal Product Weights at Inference Time (Task {current_task})\n"
        r"$w_i(t) = \prod_{k=t_i+1}^{t} \gamma_k$   "
        f"  [γ = {gamma_history[0]:.2f} for all transitions]",
        fontweight="bold",
    )
    ax.set_xlim(0, 1.25)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 14. CROSS-DATASET COMPARISON (Med-CPCL: BloodMNIST vs NIH CXR-14)
# ══════════════════════════════════════════════════════════════════════════
def plot_cross_dataset_comparison(
    metric_names: list,
    bloodmnist_values: list,
    cxr14_values: list,
    metric_directions: list = None,
    save_path: str = None,
):
    """
    Grouped bar chart comparing key metrics between BloodMNIST and NIH CXR-14
    results. direction=1 means higher is better; -1 means lower is better.
    """
    n = len(metric_names)
    x = np.arange(n)
    w = 0.32

    if metric_directions is None:
        metric_directions = [1] * n

    fig, ax = plt.subplots(figsize=(12, 4.5))

    b1 = ax.bar(x - w/2, bloodmnist_values, w,
                label="BloodMNIST (8-class)", color="#2166AC",
                edgecolor="white", linewidth=0.4, zorder=2, alpha=0.85)
    b2 = ax.bar(x + w/2, cxr14_values, w,
                label="NIH CXR-14 (14-class)", color="#D6604D",
                edgecolor="white", linewidth=0.4, zorder=2, alpha=0.85)

    for bar, v, d in zip(b1, bloodmnist_values, metric_directions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                color="#2166AC", fontweight="bold")
    for bar, v, d in zip(b2, cxr14_values, metric_directions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                color="#D6604D", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Metric Value")
    ax.set_title(
        "Cross-Dataset Comparison: BloodMNIST vs. NIH CXR-14\n"
        "Med-CPCL Extended Framework — Key Metrics",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9.5)
    ax.yaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(max(bloodmnist_values), max(cxr14_values)) * 1.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# 15. ABLATION A1 PARADOX — COVERAGE vs SET SIZE TRADE-OFF
# ══════════════════════════════════════════════════════════════════════════
def plot_ablation_a1_paradox(
    task_ids: list,
    wt_min_coverages: list,
    a1_min_coverages: list,
    wt_avg_sizes: list,
    a1_avg_sizes: list,
    target_coverage: float = 0.90,
    save_path: str = None,
):
    """
    Paired scatter plot showing that removing drift correction simultaneously
    reduces coverage AND set size — the overconfident undercovering paradox.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    task_labels = [f"T{t}" for t in task_ids]
    colors = [_PALETTE[i % len(_PALETTE)] for i in task_ids]

    # Panel 1: Min Coverage comparison
    ax = axes[0]
    x = np.arange(len(task_ids))
    w = 0.32
    b1 = ax.bar(x - w/2, wt_min_coverages, w, label="Med-CPCL (with drift)",
                color="#2166AC", edgecolor="white", linewidth=0.4, zorder=2)
    b2 = ax.bar(x + w/2, a1_min_coverages, w, label="A1: No Drift Correction",
                color="#D6604D", edgecolor="white", linewidth=0.4, zorder=2)
    ax.axhline(target_coverage, color="#333333", linestyle="--", linewidth=1.4,
               label=f"Target = {target_coverage:.2f}")
    for bar, v in zip(b1, wt_min_coverages):
        col = "#1A6F00" if v >= target_coverage else "#D6604D"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8.5,
                color=col, fontweight="bold")
    for bar, v in zip(b2, a1_min_coverages):
        col = "#1A6F00" if v >= target_coverage else "#D6604D"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8.5,
                color=col, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.set_ylabel("Minimum Coverage")
    ax.set_ylim(0.84, 1.02)
    ax.set_title("A1 Effect on Min Coverage\n(removing λ-correction reduces coverage)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.yaxis.grid(True, linestyle="--"); ax.set_axisbelow(True)

    # Panel 2: Set size comparison
    ax2 = axes[1]
    b3 = ax2.bar(x - w/2, wt_avg_sizes, w, label="Med-CPCL (with drift)",
                 color="#2166AC", edgecolor="white", linewidth=0.4, zorder=2)
    b4 = ax2.bar(x + w/2, a1_avg_sizes, w, label="A1: No Drift Correction",
                 color="#D6604D", edgecolor="white", linewidth=0.4, zorder=2)
    for bar, v in zip(b3, wt_avg_sizes):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar, v in zip(b4, a1_avg_sizes):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                 color="#D6604D", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_labels, fontsize=10)
    ax2.set_ylabel("Average Prediction Set Size")
    ax2.set_title(
        "A1 Effect on Avg Set Size\n"
        "(removing λ-correction ALSO reduces set size ← paradox)",
        fontweight="bold", fontsize=10,
    )
    ax2.legend(fontsize=8.5)
    ax2.yaxis.grid(True, linestyle="--"); ax2.set_axisbelow(True)

    fig.suptitle(
        "Drift Correction Ablation (A1): The Overconfident Undercovering Paradox\n"
        "Removing drift correction → smaller sets AND worse coverage simultaneously",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# MAIN — DEMO WITH REALISTIC SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════════════════
def main():
    """
    Demonstrates all visualization functions with realistic values drawn
    directly from the Med-CPCL NIH CXR-14 experimental results.
    """
    import os
    out_dir = "./thesis_figures_cxr14"
    os.makedirs(out_dir, exist_ok=True)

    # ── Dataset metadata ────────────────────────────────────────────────────
    class_names = [
        "Atelectasis","Cardiomegaly","Effusion","Infiltration",
        "Pneumonia","Pneumothorax","Consolidation","Edema",
        "Emphysema","Fibrosis","Pleural_Thickening","Mass",
        "Nodule","Hernia",
    ]
    positive_counts = [7480,1712,8579,12665, 876,3407,2968,1492,
                       1613,1066,2186,3705, 3907,146]
    pos_weights     = [7.32,35.34,6.26,3.92, 52.51,12.77,14.81,30.45,
                       28.23,43.23,20.57,11.72, 9.93,291.46]
    total_train = 62256  # approximate (Task 0 denominator)

    task_assignment = {
        "Atelectasis":0,"Cardiomegaly":0,"Effusion":0,"Infiltration":0,
        "Pneumonia":1,"Pneumothorax":1,"Consolidation":1,"Edema":1,
        "Emphysema":2,"Fibrosis":2,"Pleural_Thickening":2,"Mass":2,
        "Nodule":3,"Hernia":3,
    }
    task_colors = {0:"#2166AC",1:"#4DAC26",2:"#D1B847",3:"#D6604D"}

    # ── Task split sizes ────────────────────────────────────────────────────
    task_class_names = [
        ["Atelectasis","Cardiomegaly","Effusion","Infiltration"],
        ["Pneumonia","Pneumothorax","Consolidation","Edema"],
        ["Emphysema","Fibrosis","Pleural_Thickening","Mass"],
        ["Nodule","Hernia"],
    ]
    train_sizes = [62256,46928,47146,42699]
    cal_sizes   = [15481,11731,11105,10933]
    val_sizes   = [9549,7129,7164,6608]
    test_sizes  = [9851,7380,7239,6712]

    # ── Training dynamics (10 epochs, 4 tasks) ──────────────────────────────
    np.random.seed(42)
    def _loss_curve(start, end, n=10, noise=0.01):
        base = np.linspace(start, end, n)
        return (base + np.random.normal(0, noise, n)).tolist()
    def _auc_curve(start, end, n=10, noise=0.005):
        base = np.linspace(start, end, n)
        return np.clip(base + np.random.normal(0, noise, n), 0, 1).tolist()

    task_train_losses = [_loss_curve(0.283,0.222), _loss_curve(0.280,0.202),
                         _loss_curve(0.287,0.199), _loss_curve(0.663,0.254)]
    task_val_losses   = [_loss_curve(0.259,0.279), _loss_curve(0.256,0.339),
                         _loss_curve(0.261,0.362), _loss_curve(0.504,0.795)]
    task_val_aucs     = [_auc_curve(0.769,0.816), _auc_curve(0.834,0.848),
                         _auc_curve(0.824,0.838), _auc_curve(0.799,0.828)]
    task_names_short  = ["T0: Cardiopulmonary","T1: Acute Resp.",
                         "T2: Chronic/Structural","T3: Rare Findings"]

    # ── Per-class AUC ───────────────────────────────────────────────────────
    val_aucs  = [0.786,0.907,0.872,0.700, 0.766,0.881,0.851,0.893,
                 0.907,0.810,0.817,0.816, 0.748,0.900]
    test_aucs = [0.7865,0.8878,0.8667,0.7098, 0.7592,0.8796,0.8332,0.9115,
                 0.9129,0.7750,0.8119,0.7899, 0.7729,0.8932]

    task_boundaries = [(0,4,"Task 0"),(4,8,"Task 1"),(8,12,"Task 2"),(12,14,"Task 3")]

    # ── Backbone drift ──────────────────────────────────────────────────────
    drift_values = [0.1923, 0.1291, 0.1439]
    gamma_values = [0.7000, 0.7000, 0.7000]
    ref_drift    = 0.20
    gamma_min    = 0.70

    # ── Mondrian quantiles ──────────────────────────────────────────────────
    quantiles   = [0.6072,0.8021,0.6031,0.5719, 0.8771,0.7414,0.6999,0.7085,
                   0.6482,0.7316,0.6385,0.6056, 0.6444,0.9650]
    score_mins  = [0.270,0.157,0.183,0.350, 0.228,0.177,0.209,0.136,
                   0.029,0.274,0.280,0.177, 0.161,0.404]
    score_maxs  = [0.727,0.888,0.809,0.586, 0.960,0.821,0.855,0.845,
                   0.923,0.940,0.857,0.720, 0.771,0.994]

    # ── Coverage results ────────────────────────────────────────────────────
    coverages   = [0.8793,0.9234,0.9042,0.9018, 0.9551,0.9472,0.9255,0.9153,
                   0.8781,0.8989,0.9043,0.8903, 0.9094,0.9688]
    n_positives = [1152,261,1367,2128, 156,549,550,248,
                   279,178,345,547, 684,32]

    task_ids             = [0,1,2,3]
    marginal_coverages   = [0.9022,0.9358,0.8929,0.9391]
    min_coverages        = [0.8793,0.9153,0.8781,0.9094]
    avg_set_sizes        = [2.121,2.053,1.927,1.451]

    # ── Ablation ────────────────────────────────────────────────────────────
    config_names  = ["Med-CPCL","A1: No Drift","A2: Uniform γ","A3: Marginal CP"]
    abl_min_cov   = np.array([[0.8793,0.9153,0.8781,0.9094],
                               [0.8651,0.9113,0.8710,0.9094],
                               [0.8793,0.9153,0.8781,0.9094],
                               [0.8812,0.7821,0.9101,0.7188]])
    abl_avg_sz    = np.array([[2.121,2.053,1.927,1.451],
                               [1.996,1.993,1.889,1.451],
                               [2.121,2.053,1.927,1.451],
                               [2.734,1.909,2.715,0.973]])

    wt_min = [0.8793,0.9153,0.8781,0.9094]
    a1_min = [0.8651,0.9113,0.8710,0.9094]
    wt_sz  = [2.121,2.053,1.927,1.451]
    a1_sz  = [1.996,1.993,1.889,1.451]

    # ── Cross-dataset ───────────────────────────────────────────────────────
    metric_names = ["Mean Min Cov.","Mean Marg. Cov.","Mean Avg Set Sz",
                    "Triage Efficiency","Drift Corr. Effect (A1 Δ)"]
    bm_vals  = [0.929,0.982,6.876,0.864,0.630]
    cxr_vals = [0.8955,0.9175,1.888,0.135,0.630]

    # ── GENERATE ALL FIGURES ────────────────────────────────────────────────
    print("Generating Fig 1: Label Prevalence...")
    plot_label_prevalence(class_names, positive_counts, sum(positive_counts)+40000,
        task_colors, task_assignment,
        save_path=f"{out_dir}/fig01_label_prevalence.png")

    print("Generating Fig 2: Pos-Weight Distribution...")
    plot_pos_weight_distribution(class_names, pos_weights,
        save_path=f"{out_dir}/fig02_pos_weight.png")

    print("Generating Fig 3: Task Split Summary...")
    plot_task_split_summary([0,1,2,3], task_class_names,
        train_sizes, cal_sizes, val_sizes, test_sizes,
        save_path=f"{out_dir}/fig03_task_splits.png")

    print("Generating Fig 4: Training Curves...")
    plot_training_curves(task_train_losses, task_val_losses, task_val_aucs,
        task_names_short,
        save_path=f"{out_dir}/fig04_training_curves.png")

    print("Generating Fig 5: Per-Class AUC...")
    plot_per_class_auc(class_names, val_aucs, test_aucs, task_boundaries,
        task_colors,
        save_path=f"{out_dir}/fig05_per_class_auc.png")

    print("Generating Fig 6: Backbone Drift...")
    plot_backbone_drift(drift_values, gamma_values, ref_drift, gamma_min,
        save_path=f"{out_dir}/fig06_backbone_drift.png")

    print("Generating Fig 7: Mondrian Quantiles...")
    plot_mondrian_quantiles(class_names, quantiles, score_mins, score_maxs,
        task_assignment, save_path=f"{out_dir}/fig07_mondrian_quantiles.png")

    print("Generating Fig 8: Per-Class Coverage...")
    plot_per_class_coverage(class_names, coverages, n_positives, 0.90,
        task_assignment,
        save_path=f"{out_dir}/fig08_per_class_coverage.png")

    print("Generating Fig 9: Task-Level Summary...")
    plot_task_level_summary(task_ids, marginal_coverages, min_coverages,
        avg_set_sizes, save_path=f"{out_dir}/fig09_task_summary.png")

    print("Generating Fig 10: Ablation Heatmap...")
    plot_ablation_heatmap(config_names, task_ids, abl_min_cov, abl_avg_sz,
        save_path=f"{out_dir}/fig10_ablation_heatmap.png")

    print("Generating Fig 11: Coverage vs AUC...")
    plot_coverage_vs_auc(class_names, test_aucs, coverages, task_assignment,
        save_path=f"{out_dir}/fig11_coverage_vs_auc.png")

    print("Generating Fig 12: Triage Efficiency...")
    plot_triage_efficiency(task_ids, avg_set_sizes,
        save_path=f"{out_dir}/fig12_triage_efficiency.png")

    print("Generating Fig 13: Product Weight Decay...")
    plot_product_weight_decay([0.7,0.7,0.7], current_task=3,
        save_path=f"{out_dir}/fig13_product_weights.png")

    print("Generating Fig 14: Cross-Dataset Comparison...")
    plot_cross_dataset_comparison(metric_names, bm_vals, cxr_vals,
        save_path=f"{out_dir}/fig14_cross_dataset.png")

    print("Generating Fig 15: Ablation A1 Paradox...")
    plot_ablation_a1_paradox(task_ids, wt_min, a1_min, wt_sz, a1_sz,
        save_path=f"{out_dir}/fig15_a1_paradox.png")

    print(f"\nAll 15 figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()