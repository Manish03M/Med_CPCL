# experiments/run_medcpcl.py
# Phase 9: Med-CPCL full pipeline
#
# Research Validation Steps 3-5:
#   Step 3: Apply proposed method
#   Step 4: Demonstrate improvement (coverage restored, smaller sets)
#   Step 5: Comparative analysis vs Standard CP

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from datetime import datetime

from config import (NUM_TASKS, ALPHA, DEVICE, CKPT_DIR, TABLES_DIR, SEED,
                    LR, WEIGHT_DECAY, NUM_EPOCHS)
from models.model import build_model
from training.train import Trainer, evaluate_all_tasks
from training.replay_buffer import ReplayBuffer
from training.plugin import MedCPCLPlugin
from conformal.scoring import ScoreMemory
from conformal.weighted_cp import WeightedCP
from conformal.conformal import StandardCP
from data.data_loader import get_task_loaders
from experiments.run_finetuning import compute_metrics
from experiments.run_replay import ERTrainer

torch.manual_seed(SEED)
np.random.seed(SEED)

TARGET = 1 - ALPHA


def run_medcpcl():
    print("=" * 60)
    print("  PHASE 9: Med-CPCL (Proposed Method)")
    print("  Weighted CP + Drift Compensation + Mondrian CP")
    print(f"  Target coverage: {TARGET:.0%}  |  Gamma: 0.9")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────────────────
    model         = build_model()
    replay_buffer = ReplayBuffer()
    score_memory  = ScoreMemory()
    plugin        = MedCPCLPlugin(score_memory)
    trainer       = ERTrainer(model, replay_buffer)
    weighted_cp   = WeightedCP(score_memory, use_mondrian=True)
    standard_cp   = StandardCP(alpha=ALPHA, scoring="aps")  # for comparison

    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    test_loaders = []

    # Per-task CP results: {task_id: {standard: {...}, weighted: {...}}}
    cp_results = {}

    for task_id in range(NUM_TASKS):
        print(f"\n{'='*50}")
        print(f"  [Task {task_id}]")
        print(f"{'='*50}")

        train_loader, cal_loader, test_loader = get_task_loaders(task_id)
        test_loaders.append(test_loader)

        # 1. Train
        trainer.train_task(task_id, train_loader, num_epochs=NUM_EPOCHS)

        # 2. Run plugin: update score memory + drift compensation
        print(f"\n  Running Med-CPCL plugin for Task {task_id}...")
        plugin.after_training_task(model, cal_loader, task_id)

        # 3. Calibrate both CP methods
        weighted_cp.calibrate(current_task=task_id)
        standard_cp.calibrate(model, cal_loader)

        # 4. Evaluate accuracy
        print(f"\n  Accuracy on all tasks seen so far:")
        results = evaluate_all_tasks(model, test_loaders)
        for j, acc in results.items():
            acc_matrix[task_id][j] = acc
            print(f"    Task {j}: {acc:.4f}")

        # 5. Evaluate CP on all tasks seen so far
        print(f"\n  CP Evaluation (Standard vs Weighted):")
        print(f"  {'Task':<6} {'Std_Cov':>9} {'Std_Sz':>8} "
              f"{'Wt_Cov':>9} {'Wt_Sz':>8} {'Improvement':>12}")
        print("  " + "-" * 58)

        cp_results[task_id] = {}
        for j in range(task_id + 1):
            _, _, tl = get_task_loaders(j)

            std_cov, std_sz = standard_cp.evaluate(model, tl)
            wt_cov,  wt_sz  = weighted_cp.evaluate(model, tl)
            sz_improvement  = std_sz - wt_sz

            std_flag = "OK" if std_cov >= TARGET - 0.02 else "FAIL"
            wt_flag  = "OK" if wt_cov  >= TARGET - 0.02 else "FAIL"

            print(f"  T{j:<5} {std_cov:>7.4f}{std_flag:>2}  {std_sz:>7.4f}  "
                  f"{wt_cov:>7.4f}{wt_flag:>2}  {wt_sz:>7.4f}  "
                  f"{sz_improvement:>+10.4f}")

            cp_results[task_id][j] = {
                "standard": {"coverage": std_cov, "set_size": std_sz},
                "weighted": {"coverage": wt_cov,  "set_size": wt_sz}
            }

        # 6. Print Mondrian quantiles
        weighted_cp.print_quantiles()

    # ── Final metrics ──────────────────────────────────────────────────────
    AA, BWT, FM = compute_metrics(acc_matrix)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS: Med-CPCL")
    print("=" * 60)
    print(f"  Average Accuracy (AA) : {AA:.4f}")
    print(f"  Backward Transfer (BWT): {BWT:.4f}")
    print(f"  Forgetting Measure (FM): {FM:.4f}")

    # CP comparison after final task
    print("\n  CP Comparison after Task 3 (final):")
    print(f"  {'Task':<6} {'Std_Cov':>9} {'Std_Sz':>9} "
          f"{'Wt_Cov':>9} {'Wt_Sz':>9} {'Set_Reduction%':>15}")
    print("  " + "-" * 65)
    for j in range(NUM_TASKS):
        std = cp_results[NUM_TASKS-1][j]["standard"]
        wt  = cp_results[NUM_TASKS-1][j]["weighted"]
        reduction = (std["set_size"] - wt["set_size"]) / (std["set_size"] + 1e-8) * 100
        print(f"  T{j:<5} {std['coverage']:>9.4f} {std['set_size']:>9.4f} "
              f"{wt['coverage']:>9.4f} {wt['set_size']:>9.4f} "
              f"{reduction:>13.1f}%")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,   exist_ok=True)

    results_dict = {
        "method"     : "med_cpcl",
        "timestamp"  : datetime.now().isoformat(),
        "AA"         : float(AA),
        "BWT"        : float(BWT),
        "FM"         : float(FM),
        "acc_matrix" : acc_matrix.tolist(),
        "cp_results" : {
            str(ti): {
                str(tj): v for tj, v in inner.items()
            } for ti, inner in cp_results.items()
        }
    }

    out_path = os.path.join(TABLES_DIR, "medcpcl_results.json")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    ckpt_path = os.path.join(CKPT_DIR, "medcpcl_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved to: {ckpt_path}")

    return results_dict


if __name__ == "__main__":
    run_medcpcl()
