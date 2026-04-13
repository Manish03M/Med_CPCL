# experiments/run_ewc.py
# Baseline 3: Elastic Weight Consolidation (EWC)
#
# Thesis role:
#   - Regularization-based CL baseline (no data storage)
#   - Computes Fisher Information to protect important weights
#   - Expected to perform between FT and ER

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import json
from copy import deepcopy
from datetime import datetime

from config import (NUM_TASKS, NUM_EPOCHS, DEVICE, CKPT_DIR,
                    TABLES_DIR, SEED, LR, WEIGHT_DECAY, EWC_LAMBDA)
from models.model import build_model
from training.train import Trainer, evaluate_all_tasks, train_one_epoch
from data.data_loader import get_task_loaders
from experiments.run_finetuning import compute_metrics

torch.manual_seed(SEED)
np.random.seed(SEED)


class EWCTrainer(Trainer):
    """
    EWC trainer.

    After each task:
      1. Compute Fisher Information F_i for each parameter
      2. Store optimal weights theta*_i
      3. Add penalty: lambda * sum_i F_i * (theta_i - theta*_i)^2
         to the loss during subsequent tasks
    """

    def __init__(self, model, ewc_lambda=EWC_LAMBDA,
                 lr=LR, weight_decay=WEIGHT_DECAY):
        super().__init__(model, lr, weight_decay)
        self.ewc_lambda = ewc_lambda
        # List of (fisher_dict, optimal_params_dict) per completed task
        self.ewc_data   = []

    def _compute_fisher(self, loader, n_samples=200):
        """
        Estimate diagonal Fisher Information via squared gradients.
        Uses n_samples from loader for efficiency.
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters()
                  if p.requires_grad}

        count = 0
        for x, y in loader:
            if count >= n_samples:
                break
            x = x.to(self.device)
            y = y.squeeze().long().to(self.device)

            self.model.zero_grad()
            logits, _ = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

            count += x.size(0)

        # Normalise
        for n in fisher:
            fisher[n] /= min(count, n_samples)

        return fisher

    def _ewc_penalty(self):
        """Compute EWC regularisation term across all previous tasks."""
        penalty = torch.tensor(0.0, device=self.device)
        for fisher, p_star in self.ewc_data:
            for n, p in self.model.named_parameters():
                if n in fisher and p.requires_grad:
                    penalty += (fisher[n] * (p - p_star[n]) ** 2).sum()
        return self.ewc_lambda * penalty

    def _run_epoch(self, loader):
        """CE loss + EWC penalty."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.squeeze().long().to(self.device)

            self.optimizer.zero_grad()
            logits, _ = self.model(x)
            ce_loss  = self.criterion(logits, y)
            ewc_loss = self._ewc_penalty() if self.ewc_data else torch.tensor(0.0)
            loss     = ce_loss + ewc_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y).sum().item()
            total      += x.size(0)

        return total_loss / total, correct / total

    def after_task(self, task_id, loader):
        """Compute and store Fisher + optimal weights for this task."""
        print(f"  Computing Fisher Information for Task {task_id}...")
        fisher  = self._compute_fisher(loader)
        p_star  = {n: p.detach().clone()
                   for n, p in self.model.named_parameters()
                   if p.requires_grad}
        self.ewc_data.append((fisher, p_star))

        # Diagnostic: mean Fisher magnitude
        mean_f = np.mean([fisher[n].mean().item() for n in fisher])
        print(f"  Mean Fisher magnitude: {mean_f:.6f}")


def run_ewc():
    print("=" * 60)
    print("  BASELINE 3: Elastic Weight Consolidation (EWC)")
    print(f"  Lambda: {EWC_LAMBDA}")
    print("  Expected: between FT and ER performance")
    print("=" * 60)

    model   = build_model()
    trainer = EWCTrainer(model)

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
    print("  RESULTS: EWC")
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
        "method"    : "ewc",
        "timestamp" : datetime.now().isoformat(),
        "AA"        : float(AA),
        "BWT"       : float(BWT),
        "FM"        : float(FM),
        "acc_matrix": acc_matrix.tolist()
    }

    out_path = os.path.join(TABLES_DIR, "ewc_results.json")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    ckpt_path = os.path.join(CKPT_DIR, "ewc_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved to: {ckpt_path}")

    return results_dict


if __name__ == "__main__":
    run_ewc()
