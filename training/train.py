# training/train.py
# Core training engine reused by ALL experiments (FT, ER, EWC, Med-CPCL).
#
# Thesis mapping:
#   - train_one_epoch()  : standard supervised training step
#   - evaluate()         : per-task accuracy measurement
#   - Trainer class      : wraps model + optimizer, called by each experiment

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import DEVICE, LR, WEIGHT_DECAY, NUM_EPOCHS, NUM_TASKS


def train_one_epoch(model, loader, optimizer, criterion, device=DEVICE):
    """One epoch of standard supervised training. Returns avg loss."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.squeeze().long().to(device)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += x.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, device=DEVICE):
    """Evaluate accuracy on a single loader. Returns (accuracy, avg_loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.squeeze().long().to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += x.size(0)

    return correct / total, total_loss / total


def evaluate_all_tasks(model, test_loaders, device=DEVICE):
    """
    Evaluate model on all tasks seen so far.
    Returns dict: {task_id: accuracy}
    Used to compute Backward Transfer (BWT) and Forgetting.
    """
    results = {}
    for t, loader in enumerate(test_loaders):
        acc, _ = evaluate(model, loader, device)
        results[t] = acc
    return results


class Trainer:
    """
    Base trainer. Subclassed by ER, EWC trainers in later phases.
    Phase 4 (FT) uses this directly with no modification.
    """

    def __init__(self, model, lr=LR, weight_decay=WEIGHT_DECAY):
        self.model     = model
        self.device    = DEVICE
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )
        # History for logging
        self.history = []

    def train_task(self, task_id: int, train_loader: DataLoader,
                   num_epochs: int = NUM_EPOCHS, verbose: bool = True):
        """Train on a single task for num_epochs. Calls before/after hooks."""
        self.before_task(task_id, train_loader)

        if verbose:
            print(f"  Training Task {task_id}  |  "
                  f"{len(train_loader.dataset)} samples  |  "
                  f"{num_epochs} epochs")

        for epoch in range(num_epochs):
            loss, acc = self._run_epoch(train_loader)
            self.scheduler.step()
            if verbose and (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1:02d}/{num_epochs}  "
                      f"loss={loss:.4f}  acc={acc:.4f}")

        self.after_task(task_id, train_loader)

    def _run_epoch(self, loader):
        """Default epoch = standard CE. Overridden in EWC trainer."""
        return train_one_epoch(
            self.model, loader, self.optimizer, self.criterion, self.device
        )

    # ── Hooks for subclasses ───────────────────────────────────────────────
    def before_task(self, task_id, loader):
        """Called before training each task. Override in subclasses."""
        pass

    def after_task(self, task_id, loader):
        """Called after training each task. Override in subclasses."""
        pass
