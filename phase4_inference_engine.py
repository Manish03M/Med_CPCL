"""
Phase 4: Mondrian Conformal Calibration + Inference Engine
Med-CPCL Extended Framework — NIH Chest X-ray 14

Constructs per-class prediction sets C(X) = {c : s_APS(X,c) ≤ q̂_c}
and evaluates coverage + triage efficiency on all test splits.
"""

import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

# Import prior phases
from phase1_data_pipeline import build_cil_dataloaders, TASK_PARTITION, SEED
from phase2_backbone_training import (
    DualOutputResNet50,
    DEVICE,
    CHECKPOINT_DIR as PHASE2_CKPT_DIR,
    GLOBAL_LABEL_MAP,
)
from phase3_dynamic_gamma_controller import (
    DynamicGammaController,
    ScoreMemory,
    ScoreMemoryEntry,
    weighted_quantile,
    compute_aps_scores_multilabel,
    REF_DRIFT, GAMMA_MIN, ETA, ALPHA,
    SCORE_MEMORY_SIZE_PER_CLASS,
    PHASE3_DIR,
)

torch.manual_seed(SEED)

# ── Output Paths ───────────────────────────────────────────────────────────
PHASE4_DIR = Path("./results/phase4")
PHASE4_DIR.mkdir(parents=True, exist_ok=True)

# All 14 class names in global order
ALL_CLASSES = []
for labels in TASK_PARTITION.values():
    ALL_CLASSES.extend(labels)


# ── Restore Score Memory from Phase 3 ─────────────────────────────────────
def load_score_memory_and_controller() -> tuple[ScoreMemory, DynamicGammaController]:
    """
    Reconstructs the ScoreMemory and DynamicGammaController
    from the Phase 3 saved files.
    """
    print("[Phase 4] Loading Score Memory and Controller from Phase 3...")
    saved = torch.load(PHASE3_DIR / "score_memory.pt", map_location="cpu")
    ctrl_data = saved["controller"]

    # Rebuild controller with saved state
    controller = DynamicGammaController(
        ref_drift=ctrl_data["ref_drift"],
        gamma_min=ctrl_data["gamma_min"],
        eta=ctrl_data["eta"],
    )
    controller.drift_history = ctrl_data["drift_history"]
    controller.gamma_history = ctrl_data["gamma_history"]

    # Rebuild score memory
    score_memory = ScoreMemory(max_per_class=SCORE_MEMORY_SIZE_PER_CLASS)
    for cls_name, data in saved["memory"].items():
        scores   = data["scores"]    # list of floats
        task_ids = data["task_ids"]  # list of ints
        latents  = data["latents"]   # Tensor [N, 2048]

        entries = []
        for i in range(len(scores)):
            entries.append(ScoreMemoryEntry(
                score=scores[i],
                latent=latents[i],
                task_id=task_ids[i],
                cls_name=cls_name,
            ))
        score_memory.memory[cls_name] = entries

    print(f"  Loaded {score_memory.total_entries} entries across "
          f"{len(score_memory.memory)} classes")
    print(f"  Controller γ history: {controller.gamma_history}")
    print(f"  Controller drift history: {[round(d,4) for d in controller.drift_history]}")
    return score_memory, controller


# ── Quantile Computation ───────────────────────────────────────────────────
def compute_all_quantiles(
    score_memory: ScoreMemory,
    controller:   DynamicGammaController,
    current_task: int,
    alpha:        float = ALPHA,
    use_weights:  bool  = True,
) -> dict[str, float]:
    """
    Computes Mondrian weighted quantile q̂_c for every class in Score Memory.

    Args:
        score_memory: populated ScoreMemory object
        controller:   DynamicGammaController with γ history
        current_task: task index at inference time
        alpha:        significance level (0.10 → 90% coverage)
        use_weights:  if False, uses uniform weights (static CP baseline)

    Returns:
        quantiles: {cls_name: q̂_c}
    """
    quantiles = {}
    for cls_name in ALL_CLASSES:
        scores, weights = score_memory.get_scores_and_weights(
            cls_name, current_task, controller
        )
        if len(scores) == 0:
            quantiles[cls_name] = 1.0   # no data → degenerate
            continue

        if not use_weights:
            # Static CP: uniform weights (baseline)
            weights = np.ones(len(scores)) / len(scores)

        quantiles[cls_name] = weighted_quantile(scores, weights, level=1.0 - alpha)

    return quantiles


# ── Inference Engine ──────────────────────────────────────────────────────
def load_task_model(task_id: int) -> DualOutputResNet50:
    """Loads the best checkpoint for a given task."""
    ckpt_path = PHASE2_CKPT_DIR / f"task{task_id}_best.pt"
    ckpt      = torch.load(ckpt_path, map_location=DEVICE)
    n_classes = len(TASK_PARTITION[task_id])
    model     = DualOutputResNet50(num_classes=n_classes).to(DEVICE)
    model.features.load_state_dict(ckpt["backbone"])
    model.classifier.load_state_dict(ckpt["classifier"])
    model.eval()
    return model


@torch.no_grad()
def run_inference_on_loader(
    model:        DualOutputResNet50,
    loader,
    task_id:      int,
    task_classes: list[str],
    quantiles:    dict[str, float],
    device:       torch.device,
) -> dict:
    """
    Runs the full Med-CPCL inference pipeline on one test DataLoader.

    For each sample:
        1. Forward pass → (logits, latent)
        2. Per-class APS score: s_c = 1 − σ(logit_c)
        3. Prediction set: C(x) = {c : s_c ≤ q̂_c}
        4. Coverage check: Y ∈ C(x)?  (per positive class)

    Returns:
        results dict with per-class coverage and set-size metrics
    """
    # Per-class trackers
    n_positive_samples = defaultdict(int)   # true positives seen
    n_covered          = defaultdict(int)   # true positives covered by set
    all_set_sizes      = []                 # prediction set size per image
    all_triage_eff     = []                 # set_size / 14 per image

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B      = images.size(0)

        logits, _ = model(images)
        probs     = torch.sigmoid(logits)   # [B, n_task_classes]
        scores    = 1.0 - probs             # APS scores [B, n_task_classes]

        # Build prediction set for each sample
        for b in range(B):
            pred_set  = []
            set_size  = 0

            for local_idx, cls_name in enumerate(task_classes):
                s_bc  = scores[b, local_idx].item()
                q_hat = quantiles.get(cls_name, 1.0)

                if s_bc <= q_hat:
                    pred_set.append(cls_name)
                    set_size += 1

                # Coverage check for positive (true) labels
                if labels[b, local_idx].item() == 1.0:
                    n_positive_samples[cls_name] += 1
                    if cls_name in pred_set:
                        n_covered[cls_name] += 1

            all_set_sizes.append(set_size)
            all_triage_eff.append(set_size / 14.0)

    # Per-class coverage
    per_class_coverage = {}
    for cls_name in task_classes:
        n_pos = n_positive_samples[cls_name]
        if n_pos > 0:
            per_class_coverage[cls_name] = n_covered[cls_name] / n_pos
        else:
            per_class_coverage[cls_name] = float("nan")

    # Aggregate metrics
    valid_coverages = [v for v in per_class_coverage.values()
                       if not math.isnan(v)]

    return {
        "per_class_coverage":     per_class_coverage,
        "marginal_coverage":      float(np.mean(valid_coverages)) if valid_coverages else 0.0,
        "min_coverage":           float(np.min(valid_coverages))  if valid_coverages else 0.0,
        "avg_set_size":           float(np.mean(all_set_sizes)),
        "triage_efficiency":      float(np.mean(all_triage_eff)),
        "n_positive_samples":     dict(n_positive_samples),
        "n_covered":              dict(n_covered),
        "total_test_images":      len(all_set_sizes),
    }


# ── Main Phase 4 Pipeline ─────────────────────────────────────────────────
def run_phase4(batch_size: int = 32, num_workers: int = 4) -> dict:

    print("\n" + "="*65)
    print("  Phase 4: Mondrian Conformal Calibration + Inference Engine")
    print("="*65)
    print(f"  Target coverage:  {int((1-ALPHA)*100)}%  (α = {ALPHA})")
    print(f"  Evaluation:       per-class coverage + Triage Efficiency")
    print(f"  Comparison:       Static CP  vs  Dynamic-γ Med-CPCL")
    print("="*65 + "\n")

    # ── 1. Load Score Memory and Controller from Phase 3 ─────────────────
    score_memory, controller = load_score_memory_and_controller()

    # ── 2. Build DataLoaders ──────────────────────────────────────────────
    print("\n[2/4] Building DataLoaders...")
    task_data = build_cil_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        verify_images=False,
    )

    # ── 3. Compute quantiles (both weighted and static) ───────────────────
    print("\n[3/4] Computing Mondrian Quantiles...")
    # After all 4 tasks, current_task = 3
    CURRENT_TASK = 3

    quantiles_weighted = compute_all_quantiles(
        score_memory, controller, current_task=CURRENT_TASK,
        alpha=ALPHA, use_weights=True,
    )
    quantiles_static = compute_all_quantiles(
        score_memory, controller, current_task=CURRENT_TASK,
        alpha=ALPHA, use_weights=False,
    )

    print("\n  All-class quantile comparison (Weighted vs Static):")
    print(f"  {'Class':<22} {'q̂ Weighted':>12} {'q̂ Static':>10} {'Δ':>8}")
    print("  " + "─"*56)
    for cls_name in ALL_CLASSES:
        qw = quantiles_weighted[cls_name]
        qs = quantiles_static[cls_name]
        delta = qw - qs
        flag = "  ← HIGH" if qw > 0.90 else ""
        print(f"  {cls_name:<22} {qw:>12.4f} {qs:>10.4f} {delta:>+8.4f}{flag}")

    # ── 4. Evaluate on all 4 task test sets ───────────────────────────────
    print("\n[4/4] Running inference on all test sets...\n")

    all_results = {}

    for task_id in range(4):
        task_classes = TASK_PARTITION[task_id]
        print(f"  ── Task {task_id}: {task_classes} ──")

        model      = load_task_model(task_id)
        test_loader = task_data[task_id]["test"]

        # Weighted (Med-CPCL)
        results_weighted = run_inference_on_loader(
            model, test_loader, task_id, task_classes,
            quantiles_weighted, DEVICE,
        )

        # Static (baseline CP)
        results_static = run_inference_on_loader(
            model, test_loader, task_id, task_classes,
            quantiles_static, DEVICE,
        )

        all_results[task_id] = {
            "weighted": results_weighted,
            "static":   results_static,
        }

        # Print per-class coverage table
        print(f"\n  {'Class':<22} {'N+':>6} {'Wt Cov':>9} {'St Cov':>9} "
              f"{'Target':>8} {'Wt OK?':>7}")
        print("  " + "─"*65)
        for cls_name in task_classes:
            wt_cov = results_weighted["per_class_coverage"].get(cls_name, float("nan"))
            st_cov = results_static["per_class_coverage"].get(cls_name,   float("nan"))
            n_pos  = results_weighted["n_positive_samples"].get(cls_name, 0)
            target = 1.0 - ALPHA
            ok_wt  = "✅" if (not math.isnan(wt_cov) and wt_cov >= target) else "❌"
            print(f"  {cls_name:<22} {n_pos:>6} "
                  f"{wt_cov:>9.4f} {st_cov:>9.4f} "
                  f"{target:>8.2f} {ok_wt:>7}")

        print(f"\n  Summary Task {task_id}:")
        print(f"    Weighted Med-CPCL — "
              f"Marginal Cov={results_weighted['marginal_coverage']:.4f}  "
              f"Min Cov={results_weighted['min_coverage']:.4f}  "
              f"Avg Set Size={results_weighted['avg_set_size']:.3f}  "
              f"Triage Eff={results_weighted['triage_efficiency']:.4f}")
        print(f"    Static CP Baseline  — "
              f"Marginal Cov={results_static['marginal_coverage']:.4f}  "
              f"Min Cov={results_static['min_coverage']:.4f}  "
              f"Avg Set Size={results_static['avg_set_size']:.3f}  "
              f"Triage Eff={results_static['triage_efficiency']:.4f}")
        print()

    # ── 5. Cross-task summary table ───────────────────────────────────────
    print("\n" + "="*65)
    print("  CROSS-TASK SUMMARY: Weighted Med-CPCL vs Static CP")
    print("="*65)
    print(f"  {'Task':<6} {'Method':<14} {'Marg.Cov':>10} "
          f"{'Min.Cov':>9} {'AvgSetSz':>10} {'TriageEff':>11}")
    print("  " + "─"*62)

    wt_min_covs, st_min_covs = [], []
    wt_set_sizes, st_set_sizes = [], []

    for task_id in range(4):
        for method, key in [("Med-CPCL", "weighted"), ("Static CP", "static")]:
            r = all_results[task_id][key]
            safety_icon = "✅" if r["min_coverage"] >= 1-ALPHA else "❌"
            print(f"  T{task_id:<5} {method:<14} "
                  f"{r['marginal_coverage']:>10.4f} "
                  f"{r['min_coverage']:>9.4f} "
                  f"{r['avg_set_size']:>10.3f} "
                  f"{r['triage_efficiency']:>11.4f}  {safety_icon}")

            if key == "weighted":
                wt_min_covs.append(r["min_coverage"])
                wt_set_sizes.append(r["avg_set_size"])
            else:
                st_min_covs.append(r["min_coverage"])
                st_set_sizes.append(r["avg_set_size"])
        print("  " + "─"*62)

    print(f"\n  Overall (mean across tasks):")
    print(f"    Med-CPCL  — Mean Min Coverage: {np.mean(wt_min_covs):.4f}  "
          f"Mean Avg Set Size: {np.mean(wt_set_sizes):.3f}")
    print(f"    Static CP — Mean Min Coverage: {np.mean(st_min_covs):.4f}  "
          f"Mean Avg Set Size: {np.mean(st_set_sizes):.3f}")

    target_pass_wt = sum(1 for c in wt_min_covs if c >= 1-ALPHA)
    target_pass_st = sum(1 for c in st_min_covs if c >= 1-ALPHA)
    print(f"\n  Tasks passing ≥{int((1-ALPHA)*100)}% min coverage:")
    print(f"    Med-CPCL:  {target_pass_wt}/4")
    print(f"    Static CP: {target_pass_st}/4")

    # ── 6. Save results ───────────────────────────────────────────────────
    serialisable = {}
    for task_id, res in all_results.items():
        serialisable[str(task_id)] = {
            "weighted": {
                "marginal_coverage":  round(res["weighted"]["marginal_coverage"], 4),
                "min_coverage":       round(res["weighted"]["min_coverage"], 4),
                "avg_set_size":       round(res["weighted"]["avg_set_size"], 4),
                "triage_efficiency":  round(res["weighted"]["triage_efficiency"], 4),
                "per_class_coverage": {
                    k: round(v, 4) if not math.isnan(v) else None
                    for k, v in res["weighted"]["per_class_coverage"].items()
                },
            },
            "static": {
                "marginal_coverage": round(res["static"]["marginal_coverage"], 4),
                "min_coverage":      round(res["static"]["min_coverage"], 4),
                "avg_set_size":      round(res["static"]["avg_set_size"], 4),
                "triage_efficiency": round(res["static"]["triage_efficiency"], 4),
                "per_class_coverage": {
                    k: round(v, 4) if not math.isnan(v) else None
                    for k, v in res["static"]["per_class_coverage"].items()
                },
            },
        }
    serialisable["quantiles_weighted"] = {
        k: round(v, 4) for k, v in quantiles_weighted.items()
    }
    serialisable["quantiles_static"] = {
        k: round(v, 4) for k, v in quantiles_static.items()
    }

    results_path = PHASE4_DIR / "phase4_results.json"
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  [Saved] Results → {results_path}")
    print("\n[Phase 4 COMPLETE] — Ready for Phase 5: Full Evaluation Report.\n")

    return all_results, quantiles_weighted, quantiles_static


# ── Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    all_results, q_weighted, q_static = run_phase4(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )