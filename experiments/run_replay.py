# experiments/run_replay.py
# Baseline 2: Experience Replay (ER)
#
# Thesis role:
#   - Main CL accuracy baseline
#   - Replay buffer here is the SAME mechanism feeding Score Memory in Phase 9
#   - Should significantly improve BWT vs Fine-Tuning

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime

from config import (NUM_TASKS, NUM_EPOCHS, DEVICE, CKPT_DIR,
                    TABLES_DIR, SEED, LR, WEIGHT_DECAY)
from models.model import build_model
from training.train import Trainer, evaluate_all_tasks, train_one_epoch
from training.replay_buffer import ReplayBuffer
from data.data_loader import get_task_loaders
from experiments.run_finetuning import compute_metrics

torch.manual_seed(SEED)
np.random.seed(SEED)


class ERTrainer(Trainer):
    """
    Experience Replay trainer.
    Overrides _run_epoch to mix current task data with replay buffer.
    Overrides after_task to update the replay buffer.
    """

    def __init__(self, model, replay_buffer: ReplayBuffer,
                 lr=LR, weight_decay=WEIGHT_DECAY):
        super().__init__(model, lr, weight_decay)
        self.replay_buffer  = replay_buffer
        self._current_loader = None   # set in before_task

    def before_task(self, task_id, loader):
        self._current_loader = loader

    def _run_epoch(self, loader):
        """Train on combined: current task + replay buffer."""
        combined = self.replay_buffer.get_combined_loader(loader)
        return train_one_epoch(
            self.model, combined, self.optimizer,
            self.criterion, self.device
        )

    def after_task(self, task_id, loader):
        """Update replay buffer with current task samples."""
        self.replay_buffer.update(loader, task_id)
        self.replay_buffer.summary()


def run_replay():
    print("=" * 60)
    print("  BASELINE 2: Experience Replay (ER)")
    print(f"  Buffer: 20 samples/class")
    print("  Expected: reduced forgetting vs Fine-Tuning")
    print("=" * 60)

    model          = build_model()
    replay_buffer  = ReplayBuffer()
    trainer        = ERTrainer(model, replay_buffer)

    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    test_loaders = []

    for task_id in range(NUM_TASKS):
        print(f"\n[Task {task_id}]")

        train_loader, cal_loader, test_loader = get_task_loaders(task_id)
        test_loaders.append(test_loader)

        trainer.train_task(task_id, train_loader, num_epochs=NUM_EPOCHS)

        print(f"  Evaluating all {task_id+1} tasks...")
        results = evaluate_all_tasks(model, test_loaders)
        for j, acc in results.items():
            acc_matrix[task_id][j] = acc
            print(f"    Task {j} accuracy: {acc:.4f}")

    AA, BWT, FM = compute_metrics(acc_matrix)

    print("\n" + "=" * 60)
    print("  RESULTS: Experience Replay")
    print("=" * 60)
    print(f"  Average Accuracy (AA) : {AA:.4f}")
    print(f"  Backward Transfer (BWT): {BWT:.4f}")
    print(f"  Forgetting Measure (FM): {FM:.4f}")
    print("\n  Accuracy Matrix:")
    header = "       " + "  ".join([f"T{j}" for j in range(NUM_TASKS)])
    print(header)
    for i in range(NUM_TASKS):
        row = f"  T{i} | " + "  ".join(
            [f"{acc_matrix[i][j]:.3f}" for j in range(NUM_TASKS)])
        print(row)
    print("=" * 60)

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,   exist_ok=True)

    results_dict = {
        "method"    : "experience_replay",
        "timestamp" : datetime.now().isoformat(),
        "AA"        : float(AA),
        "BWT"       : float(BWT),
        "FM"        : float(FM),
        "acc_matrix": acc_matrix.tolist()
    }

    out_path = os.path.join(TABLES_DIR, "replay_results.json")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    ckpt_path = os.path.join(CKPT_DIR, "replay_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved to: {ckpt_path}")

    return results_dict


if __name__ == "__main__":
    run_replay()
