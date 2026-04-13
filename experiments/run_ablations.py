# experiments/run_ablations.py
# Phase 11: Ablation Studies for Med-CPCL
#
# Three ablations:
#   A1: No drift compensation (drift_correction=0.0)
#   A2: No time weighting (gamma=1.0 treats all scores as equally fresh)
#   A3: Buffer size sensitivity (5, 20, 50 samples/class)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from datetime import datetime

from config import (NUM_TASKS, NUM_EPOCHS, DEVICE, TABLES_DIR, CKPT_DIR,
                    SEED, LR, WEIGHT_DECAY, ALPHA)
from models.model import build_model
from training.train import evaluate_all_tasks
from training.replay_buffer import ReplayBuffer
from training.plugin import MedCPCLPlugin
from conformal.scoring import ScoreMemory
from conformal.weighted_cp import WeightedCP
from data.data_loader import get_task_loaders
from experiments.run_finetuning import compute_metrics
from experiments.run_replay import ERTrainer

torch.manual_seed(SEED)
np.random.seed(SEED)

TARGET = 1 - ALPHA


def run_single_config(gamma=0.9, drift_correction=0.005,
                      buffer_size=20, label="default"):
    """Run one complete Med-CPCL experiment with given hyperparameters."""
    from config import SCORE_MEMORY_SIZE

    model         = build_model()
    replay_buffer = ReplayBuffer(max_per_class=buffer_size)
    score_memory  = ScoreMemory()
    plugin        = MedCPCLPlugin(score_memory,
                                   gamma=gamma,
                                   drift_correction=drift_correction)
    trainer       = ERTrainer(model, replay_buffer)
    wcp           = WeightedCP(score_memory, alpha=ALPHA,
                               gamma=gamma, use_mondrian=True)

    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    test_loaders = []
    final_cp     = {}

    for task_id in range(NUM_TASKS):
        train_loader, cal_loader, test_loader = get_task_loaders(task_id)
        test_loaders.append(test_loader)

        trainer.train_task(task_id, train_loader,
                           num_epochs=NUM_EPOCHS, verbose=False)
        plugin.after_training_task(model, cal_loader, task_id)
        wcp.calibrate(current_task=task_id)

        results = evaluate_all_tasks(model, test_loaders)
        for j, acc in results.items():
            acc_matrix[task_id][j] = acc

    # Final CP evaluation on all tasks
    for j in range(NUM_TASKS):
        _, _, tl = get_task_loaders(j)
        cov, sz  = wcp.evaluate(model, tl)
        final_cp[j] = {"coverage": cov, "set_size": sz}

    AA, BWT, FM = compute_metrics(acc_matrix)
    avg_cov  = np.mean([final_cp[j]["coverage"]  for j in range(NUM_TASKS)])
    avg_sz   = np.mean([final_cp[j]["set_size"]  for j in range(NUM_TASKS)])
    min_cov  = min(final_cp[j]["coverage"] for j in range(NUM_TASKS))

    return {
        "label"    : label,
        "gamma"    : gamma,
        "drift_cor": drift_correction,
        "buf_size" : buffer_size,
        "AA"       : float(AA),
        "BWT"      : float(BWT),
        "avg_cov"  : float(avg_cov),
        "min_cov"  : float(min_cov),
        "avg_sz"   : float(avg_sz),
        "per_task_cp": {str(j): final_cp[j] for j in range(NUM_TASKS)}
    }


def run_ablations():
    print("=" * 65)
    print("  PHASE 11: Ablation Studies")
    print("=" * 65)

    all_results = {}

    # ── Ablation 1: Drift Compensation ─────────────────────────────────────
    print("\n[A1] Drift Compensation Ablation")
    print("  Testing: with drift (0.005) vs without drift (0.0)")

    configs_A1 = [
        {"gamma": 0.9, "drift_correction": 0.005, "buffer_size": 20,
         "label": "With Drift (proposed)"},
        {"gamma": 0.9, "drift_correction": 0.0,   "buffer_size": 20,
         "label": "No Drift Compensation"},
    ]
    results_A1 = []
    for cfg in configs_A1:
        print(f"  Running: {cfg['label']}...")
        r = run_single_config(**cfg)
        results_A1.append(r)
        print(f"    avg_cov={r['avg_cov']:.4f}  min_cov={r['min_cov']:.4f}  "
              f"avg_sz={r['avg_sz']:.4f}  AA={r['AA']:.4f}")
    all_results["A1_drift"] = results_A1

    # ── Ablation 2: Time Weighting ──────────────────────────────────────────
    print("\n[A2] Time-Aware Weighting Ablation")
    print("  Testing: gamma=0.9 (proposed) vs gamma=1.0 (uniform weights)")

    configs_A2 = [
        {"gamma": 0.9, "drift_correction": 0.005, "buffer_size": 20,
         "label": "Gamma=0.9 (proposed)"},
        {"gamma": 1.0, "drift_correction": 0.005, "buffer_size": 20,
         "label": "Gamma=1.0 (no decay)"},
    ]
    results_A2 = []
    for cfg in configs_A2:
        print(f"  Running: {cfg['label']}...")
        r = run_single_config(**cfg)
        results_A2.append(r)
        print(f"    avg_cov={r['avg_cov']:.4f}  min_cov={r['min_cov']:.4f}  "
              f"avg_sz={r['avg_sz']:.4f}  AA={r['AA']:.4f}")
    all_results["A2_gamma"] = results_A2

    # ── Ablation 3: Buffer Size Sensitivity ────────────────────────────────
    print("\n[A3] Replay Buffer Size Sensitivity")
    print("  Testing: 5, 20, 50 samples/class")

    configs_A3 = [
        {"gamma": 0.9, "drift_correction": 0.005, "buffer_size": 5,
         "label": "Buffer=5"},
        {"gamma": 0.9, "drift_correction": 0.005, "buffer_size": 20,
         "label": "Buffer=20 (proposed)"},
        {"gamma": 0.9, "drift_correction": 0.005, "buffer_size": 50,
         "label": "Buffer=50"},
    ]
    results_A3 = []
    for cfg in configs_A3:
        print(f"  Running: {cfg['label']}...")
        r = run_single_config(**cfg)
        results_A3.append(r)
        print(f"    avg_cov={r['avg_cov']:.4f}  min_cov={r['min_cov']:.4f}  "
              f"avg_sz={r['avg_sz']:.4f}  AA={r['AA']:.4f}")
    all_results["A3_buffer"] = results_A3

    # ── Summary Tables ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ABLATION SUMMARY")
    print("=" * 65)

    def print_table(label, results):
        print(f"\n  {label}")
        print(f"  {'Config':<28} {'AA':>7} {'AvgCov':>8} {'MinCov':>8} {'AvgSz':>8}")
        print("  " + "-" * 63)
        for r in results:
            flag = " <-- proposed" if "proposed" in r["label"] else ""
            print(f"  {r['label']:<28} {r['AA']:>7.4f} {r['avg_cov']:>8.4f} "
                  f"{r['min_cov']:>8.4f} {r['avg_sz']:>8.4f}{flag}")

    print_table("A1: Drift Compensation",    results_A1)
    print_table("A2: Time-Aware Weighting",  results_A2)
    print_table("A3: Buffer Size",           results_A3)

    print("\n" + "=" * 65)

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(TABLES_DIR, exist_ok=True)
    out = {
        "timestamp": datetime.now().isoformat(),
        "ablations": all_results
    }
    path = os.path.join(TABLES_DIR, "ablation_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved to: {path}")

    return all_results


if __name__ == "__main__":
    run_ablations()
