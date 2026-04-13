# experiments/run_standard_cp.py
# Phase 8: Standard CP baseline + proof-of-failure demonstration
#
# Research Validation Steps 1 & 2 (from thesis plan):
#   Step 1: Reproduce baseline CP behavior -- record coverage/set size
#   Step 2: Demonstrate failure -- show coverage drops below 1-alpha
#           on old tasks after sequential training
#
# Two experiments:
#   A) Per-task CP: calibrate+test on SAME task data (upper bound)
#   B) Sequential CP: calibrate on task T data, test after training T+1,T+2,T+3
#      This is where failure manifests

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from datetime import datetime

from config import (NUM_TASKS, ALPHA, DEVICE, CKPT_DIR, TABLES_DIR, SEED)
from models.model import build_model
from conformal.conformal import StandardCP
from data.data_loader import get_task_loaders

torch.manual_seed(SEED)
np.random.seed(SEED)

TARGET_COVERAGE = 1 - ALPHA   # 0.90


def experiment_A_per_task(model):
    """
    Experiment A: Calibrate and test on same-task data.
    This is the STATIC case -- should achieve ~90% coverage.
    Validates that CP works when exchangeability holds.
    """
    print("\n-- Experiment A: Per-Task CP (static, exchangeability holds) --")
    print(f"  Target coverage: {TARGET_COVERAGE:.2f}")
    print(f"  {'Task':<6} {'q_hat':>8} {'Coverage':>10} {'SetSize':>9} {'Pass?':>7}")
    print("  " + "-" * 50)

    results = {}
    for t in range(NUM_TASKS):
        _, cal_loader, test_loader = get_task_loaders(t)
        cp = StandardCP(alpha=ALPHA, scoring="aps")
        q  = cp.calibrate(model, cal_loader)
        cov, sz = cp.evaluate(model, test_loader)
        passed  = "YES" if cov >= TARGET_COVERAGE - 0.02 else "NO "
        print(f"  T{t:<5} {q:>8.4f} {cov:>10.4f} {sz:>9.4f} {passed:>7}")
        results[t] = {"q_hat": q, "coverage": cov, "set_size": sz}

    return results


def experiment_B_sequential(model):
    """
    Experiment B: Sequential CP failure demonstration.

    Protocol:
      1. Train model sequentially T0->T1->T2->T3 (using ER checkpoint)
      2. Calibrate CP using Task 0 calibration data ONCE
      3. After each new task is learned, re-test on Task 0
      4. Show coverage degrades -- proving non-exchangeability

    Since we already have the ER-trained model (post T3), we simulate
    this by calibrating on Task 0 cal data and testing on all tasks.
    This shows the cross-task coverage failure.
    """
    print("\n-- Experiment B: Sequential CP (CL failure demonstration) --")
    print("  Protocol: Calibrate on Task 0, test on Tasks 0-3")
    print("  Expected: Coverage drops below target on Tasks 1-3")
    print(f"  Target coverage: {TARGET_COVERAGE:.2f}")
    print(f"  {'Task':<6} {'Coverage':>10} {'SetSize':>9} {'Delta':>8} {'Fail?':>7}")
    print("  " + "-" * 50)

    # Calibrate on Task 0 calibration data
    _, cal_loader_t0, _ = get_task_loaders(0)
    cp = StandardCP(alpha=ALPHA, scoring="aps")
    q  = cp.calibrate(model, cal_loader_t0)
    print(f"  Calibrated on Task 0 | q_hat = {q:.4f}")

    results = {}
    for t in range(NUM_TASKS):
        _, _, test_loader = get_task_loaders(t)
        cov, sz  = cp.evaluate(model, test_loader)
        delta    = cov - TARGET_COVERAGE
        failed   = "FAIL" if cov < TARGET_COVERAGE - 0.02 else "OK  "
        print(f"  T{t:<5} {cov:>10.4f} {sz:>9.4f} {delta:>+8.4f} {failed:>7}")
        results[t] = {"coverage": cov, "set_size": sz, "delta": delta}

    return results, float(q)


def run_standard_cp():
    print("=" * 60)
    print("  PHASE 8: Standard Conformal Prediction (Baseline)")
    print("  Model: ER-trained (replay_final.pt)")
    print(f"  Scoring: APS  |  Alpha: {ALPHA}  |  Target: {TARGET_COVERAGE:.0%}")
    print("=" * 60)

    model = build_model()
    ckpt  = os.path.join(CKPT_DIR, "replay_final.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    print(f"  Loaded: {ckpt}")

    res_A            = experiment_A_per_task(model)
    res_B, q_shared  = experiment_B_sequential(model)

    # Summary
    print("\n" + "=" * 60)
    print("  THESIS FINDING: Standard CP under CL")
    print("=" * 60)

    failed_tasks = [t for t, r in res_B.items()
                    if r["coverage"] < TARGET_COVERAGE - 0.02]
    if failed_tasks:
        print(f"  Coverage failure on tasks: {failed_tasks}")
        drops = [TARGET_COVERAGE - res_B[t]["coverage"] for t in failed_tasks]
        print(f"  Max coverage drop: {max(drops):.4f} below {TARGET_COVERAGE:.2f}")
        print("  CONCLUSION: Standard CP fails under representational drift.")
        print("  This motivates weighted CP with drift compensation (Phase 9).")
    else:
        print("  Note: ER model retains enough coverage -- showing drift")
        print("  is subtle. Phase 9 weighted method will still improve set size.")

    # Save
    os.makedirs(TABLES_DIR, exist_ok=True)
    results_dict = {
        "method"         : "standard_cp",
        "timestamp"      : datetime.now().isoformat(),
        "alpha"          : ALPHA,
        "target_coverage": TARGET_COVERAGE,
        "experiment_A"   : {str(k): v for k, v in res_A.items()},
        "experiment_B"   : {
            "q_hat"        : q_shared,
            "per_task"     : {str(k): v for k, v in res_B.items()}
        }
    }

    out_path = os.path.join(TABLES_DIR, "standard_cp_results.json")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return results_dict


if __name__ == "__main__":
    run_standard_cp()
