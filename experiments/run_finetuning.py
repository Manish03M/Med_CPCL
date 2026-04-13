# experiments/run_finetuning.py
# Baseline 1: Naive Fine-Tuning (FT)
#
# Thesis role:
#   - Proves catastrophic forgetting: Task 0 accuracy collapses after Task 3
#   - Establishes the LOWER BOUND all CL methods must beat
#   - Records per-task accuracy matrix A[i][j] = acc on task j after training task i
#
# Metrics computed:
#   - Average Accuracy (AA)
#   - Backward Transfer (BWT) -- negative = forgetting
#   - Forgetting Measure (FM)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from datetime import datetime

from config import (NUM_TASKS, NUM_EPOCHS, DEVICE, CKPT_DIR,
                    RESULTS_DIR, TABLES_DIR, SEED)
from models.model import build_model
from training.train import Trainer, evaluate_all_tasks
from data.data_loader import get_task_loaders

torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics(acc_matrix):
    """
    acc_matrix[i][j] = accuracy on task j immediately after training task i.
    Rows = after training task i. Cols = task j being evaluated.

    Returns:
        AA  : Average Accuracy after final task
        BWT : Backward Transfer (negative = forgetting)
        FM  : Forgetting Measure (max drop per task)
    """
    T = len(acc_matrix)

    # AA: mean accuracy on all tasks after training on last task
    AA = np.mean([acc_matrix[T-1][j] for j in range(T)])

    # BWT: average change in old task accuracy
    BWT = np.mean([acc_matrix[T-1][j] - acc_matrix[j][j] for j in range(T-1)])

    # FM: average of maximum forgetting per task
    FM_per_task = []
    for j in range(T-1):
        max_acc = max(acc_matrix[i][j] for i in range(j, T))
        final   = acc_matrix[T-1][j]
        FM_per_task.append(max_acc - final)
    FM = np.mean(FM_per_task)

    return AA, BWT, FM


def run_finetuning():
    print("=" * 60)
    print("  BASELINE 1: Naive Fine-Tuning")
    print("  Expected: catastrophic forgetting on early tasks")
    print("=" * 60)

    model   = build_model()
    trainer = Trainer(model)

    # acc_matrix[i][j]: accuracy on task j after training task i
    acc_matrix  = np.zeros((NUM_TASKS, NUM_TASKS))
    test_loaders = []

    for task_id in range(NUM_TASKS):
        print(f"\n[Task {task_id}]")

        # Get loaders
        train_loader, cal_loader, test_loader = get_task_loaders(task_id)
        test_loaders.append(test_loader)

        # Train on current task (no replay, no regularization)
        trainer.train_task(task_id, train_loader, num_epochs=NUM_EPOCHS)

        # Evaluate on ALL tasks seen so far
        print(f"  Evaluating all {task_id+1} tasks...")
        results = evaluate_all_tasks(model, test_loaders)

        for j, acc in results.items():
            acc_matrix[task_id][j] = acc
            print(f"    Task {j} accuracy: {acc:.4f}")

    # ── Compute & print metrics ────────────────────────────────────────────
    AA, BWT, FM = compute_metrics(acc_matrix)

    print("\n" + "=" * 60)
    print("  RESULTS: Naive Fine-Tuning")
    print("=" * 60)
    print(f"  Average Accuracy (AA) : {AA:.4f}")
    print(f"  Backward Transfer (BWT): {BWT:.4f}  (negative = forgetting)")
    print(f"  Forgetting Measure (FM): {FM:.4f}")
    print("\n  Accuracy Matrix (row=after task i, col=task j):")
    header = "       " + "  ".join([f"T{j}" for j in range(NUM_TASKS)])
    print(header)
    for i in range(NUM_TASKS):
        row = f"  T{i} | " + "  ".join([f"{acc_matrix[i][j]:.3f}" for j in range(NUM_TASKS)])
        print(row)
    print("=" * 60)

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    results_dict = {
        "method"     : "finetuning",
        "timestamp"  : datetime.now().isoformat(),
        "AA"         : float(AA),
        "BWT"        : float(BWT),
        "FM"         : float(FM),
        "acc_matrix" : acc_matrix.tolist()
    }

    out_path = os.path.join(TABLES_DIR, "finetuning_results.json")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # Save checkpoint
    ckpt_path = os.path.join(CKPT_DIR, "finetuning_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved to: {ckpt_path}")

    return results_dict


if __name__ == "__main__":
    run_finetuning()
