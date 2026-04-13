"""
Phase 2: ResNet-50 Dual-Output Backbone + Focal Loss + Experience Replay
Med-CPCL Extended Framework — NIH Chest X-ray 14
"""

import os
import time
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import roc_auc_score

# Import Phase 1 components
from phase1_data_pipeline import (
    build_cil_dataloaders,
    TASK_PARTITION,
    SEED,
)

# ── Reproducibility ────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Config ─────────────────────────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR  = Path("./checkpoints/phase2")
RESULTS_DIR     = Path("./results/phase2")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
NUM_EPOCHS_PER_TASK = 10
LR                  = 1e-4        # Lower LR for pretrained ResNet-50
WEIGHT_DECAY        = 1e-4
BATCH_SIZE          = 32
NUM_WORKERS         = 4

# Experience Replay
ER_BUFFER_SIZE_PER_CLASS = 50     # 50 samples/class (7 task classes max)

# Focal Loss
FOCAL_GAMMA = 2.0                 # Focus parameter — punishes easy negatives less

# ── Model: Dual-Output ResNet-50 ──────────────────────────────────────────
class DualOutputResNet50(nn.Module):
    """
    ResNet-50 backbone with dual output:
      - logits : B × num_task_classes  (for BCE/Focal loss)
      - latent  : B × 2048             (penultimate layer, for drift estimation)

    The classifier head is task-specific and grows as new tasks are learned.
    The backbone (self.features) is shared across all tasks.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Load ImageNet-pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Backbone = everything except the final FC layer
        # Output: B × 2048 (after global average pooling)
        self.features  = nn.Sequential(*list(resnet.children())[:-1])

        # Task-specific classifier head
        self.classifier = nn.Linear(2048, num_classes)

        # Dropout before classifier for regularisation
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor [B, 3, 224, 224]
        Returns:
            logits: Tensor [B, num_classes]  — raw (pre-sigmoid) scores
            latent: Tensor [B, 2048]         — L2-normalised feature vector
        """
        # Feature extraction
        z = self.features(x)          # [B, 2048, 1, 1]
        z = z.flatten(1)              # [B, 2048]

        # L2-normalise latent for stable drift estimation
        latent = F.normalize(z, p=2, dim=1)

        # Classification head
        logits = self.classifier(self.dropout(z))  # [B, num_classes]

        return logits, latent

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: returns only latent z (for CP calibration)."""
        with torch.no_grad():
            _, latent = self.forward(x)
        return latent


# ── Focal Loss ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Multi-label Focal Loss = -α · (1 − p_t)^γ · log(p_t)

    Combines:
      - Focal weighting (γ=2): down-weights easy well-classified examples
      - pos_weight (α): up-weights positive (disease) examples to counter imbalance

    This is the correct loss for NIH-CXR-14 where Hernia appears in only 0.34%
    of images and the standard BCE loss would learn to predict "No Hernia" always.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] — raw (pre-sigmoid) scores
            targets: [B, C] — binary multi-hot labels (float)
        Returns:
            scalar loss
        """
        # Standard BCE (with pos_weight for imbalance)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
            reduction="none",
        )  # [B, C]

        # Focal weight: (1 - p_t)^gamma
        p_t = torch.where(targets == 1,
                          torch.sigmoid(logits),
                          1 - torch.sigmoid(logits))
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * bce
        return loss.mean()


# ── Experience Replay Buffer ───────────────────────────────────────────────
class ExperienceReplayBuffer:
    """
    Reservoir-sampling replay buffer for multi-label CIL.

    For each class c, stores up to `max_per_class` samples.
    At training time, mixes replay samples with current task data.

    Storage format (in-memory tensors for GPU efficiency):
        images:  Tensor [N, 3, 224, 224]   (float32, normalised)
        labels:  Tensor [N, 14]            (float32, full 14-class vector)
        task_id: Tensor [N]                (int64)
    """

    def __init__(self, max_per_class: int = ER_BUFFER_SIZE_PER_CLASS):
        self.max_per_class = max_per_class
        self.buffer: dict[int, dict] = {}  # class_idx → {images, labels, task_ids, count}
        self.total_seen: dict[int, int] = defaultdict(int)

    def add_task_data(
        self,
        loader: DataLoader,
        task_id: int,
        task_classes: list[str],
        device: torch.device,
        global_label_map: dict,  # class_name → global index 0–13
    ) -> None:
        """
        Fills the buffer from a task's calibration DataLoader using
        reservoir sampling per class.

        Args:
            loader:          calibration DataLoader for the task
            task_id:         integer task identifier
            task_classes:    list of class names for this task
            device:          torch device
            global_label_map: maps class name to its 0–13 global index
        """
        print(f"  [ER Buffer] Collecting samples for Task {task_id}...")

        # Temporary per-class reservoirs: list of (image, full_label) tuples
        reservoirs: dict[int, list] = {
            global_label_map[cls]: [] for cls in task_classes
        }

        n_batches = len(loader)
        for batch_idx, (images, task_labels, _) in enumerate(loader):
            images = images.to(device)
            B      = images.size(0)

            # Build full 14-class label for each sample
            full_labels = torch.zeros(B, 14)
            for local_idx, cls_name in enumerate(task_classes):
                global_idx = global_label_map[cls_name]
                full_labels[:, global_idx] = task_labels[:, local_idx]

            # Reservoir sampling per class
            for b in range(B):
                for local_idx, cls_name in enumerate(task_classes):
                    if task_labels[b, local_idx].item() == 1.0:
                        g_idx = global_label_map[cls_name]
                        self.total_seen[g_idx] += 1
                        n = self.total_seen[g_idx]

                        if len(reservoirs[g_idx]) < self.max_per_class:
                            reservoirs[g_idx].append(
                                (images[b].cpu(), full_labels[b])
                            )
                        else:
                            # Reservoir sampling: replace with prob max_per_class/n
                            j = random.randint(0, n - 1)
                            if j < self.max_per_class:
                                reservoirs[g_idx][j] = (
                                    images[b].cpu(), full_labels[b]
                                )

        # Store in buffer
        for g_idx, samples in reservoirs.items():
            if not samples:
                continue
            imgs_t   = torch.stack([s[0] for s in samples])
            labels_t = torch.stack([s[1] for s in samples])
            tids_t   = torch.full((len(samples),), task_id, dtype=torch.long)

            if g_idx not in self.buffer:
                self.buffer[g_idx] = {
                    "images":   imgs_t,
                    "labels":   labels_t,
                    "task_ids": tids_t,
                }
            else:
                self.buffer[g_idx]["images"]   = torch.cat(
                    [self.buffer[g_idx]["images"],   imgs_t], dim=0
                )[-self.max_per_class:]
                self.buffer[g_idx]["labels"]   = torch.cat(
                    [self.buffer[g_idx]["labels"],   labels_t], dim=0
                )[-self.max_per_class:]
                self.buffer[g_idx]["task_ids"] = torch.cat(
                    [self.buffer[g_idx]["task_ids"], tids_t], dim=0
                )[-self.max_per_class:]

        total = sum(len(v["images"]) for v in self.buffer.values())
        print(f"  [ER Buffer] Total samples stored: {total} "
              f"across {len(self.buffer)} classes")

    def get_replay_loader(
        self,
        task_classes: list[str],
        batch_size: int,
        global_label_map: dict,
    ) -> DataLoader | None:
        """
        Returns a DataLoader over all buffered samples, with labels projected
        to the current task's class subset.
        """
        if not self.buffer:
            return None

        all_images, all_labels = [], []
        for data in self.buffer.values():
            all_images.append(data["images"])
            all_labels.append(data["labels"])

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Project full 14-class labels to current task's classes
        task_indices = torch.tensor(
            [global_label_map[cls] for cls in task_classes], dtype=torch.long
        )
        task_labels = all_labels[:, task_indices]

        dataset = TensorDataset(all_images, task_labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def size(self) -> int:
        return sum(len(v["images"]) for v in self.buffer.values())


# ── Training Functions ─────────────────────────────────────────────────────
def train_one_epoch(
    model:        DualOutputResNet50,
    loader:       DataLoader,
    replay_loader: DataLoader | None,
    optimizer:    torch.optim.Optimizer,
    criterion:    FocalLoss,
    device:       torch.device,
    epoch:        int,
    task_id:      int,
) -> dict:
    """Trains for one epoch, mixing current task data with replay samples."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    # Create iterators
    current_iter = iter(loader)
    replay_iter  = iter(replay_loader) if replay_loader else None

    # Process all current-task batches
    for images, labels, _ in current_iter:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mix with replay batch if available
        if replay_iter is not None:
            try:
                r_images, r_labels = next(replay_iter)
                r_images = r_images.to(device, non_blocking=True)
                r_labels = r_labels.to(device, non_blocking=True)
                images   = torch.cat([images, r_images], dim=0)
                labels   = torch.cat([labels, r_labels], dim=0)
            except StopIteration:
                pass  # Replay exhausted — continue with current data only

        optimizer.zero_grad()
        logits, _ = model(images)
        loss      = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stability with pretrained weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    avg_loss = total_loss / max(n_batches, 1)
    return {"loss": avg_loss}


@torch.no_grad()
def evaluate(
    model:        DualOutputResNet50,
    loader:       DataLoader,
    criterion:    FocalLoss,
    device:       torch.device,
    task_classes: list[str],
) -> dict:
    """
    Evaluates model on a DataLoader.
    Returns loss, per-class AUC-ROC, and mean AUC.
    """
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_batches  = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, _ = model(images)
        loss       = criterion(logits, labels)

        total_loss += loss.item()
        n_batches  += 1

        all_logits.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss   = total_loss / max(n_batches, 1)

    # Per-class AUC-ROC (skip classes with no positives in this split)
    per_class_auc = {}
    for i, cls_name in enumerate(task_classes):
        if all_labels[:, i].sum() > 0:
            per_class_auc[cls_name] = roc_auc_score(
                all_labels[:, i], all_logits[:, i]
            )
        else:
            per_class_auc[cls_name] = float("nan")

    valid_aucs = [v for v in per_class_auc.values() if not np.isnan(v)]
    mean_auc   = np.mean(valid_aucs) if valid_aucs else 0.0

    return {
        "loss":          avg_loss,
        "mean_auc":      mean_auc,
        "per_class_auc": per_class_auc,
    }


@torch.no_grad()
def compute_task_prototype(
    model:  DualOutputResNet50,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes the mean latent vector μ(t) over the calibration set.
    This is the global prototype used for drift estimation in Phase 3.

    Returns:
        mu: Tensor [2048] — mean L2-normalised latent vector
    """
    model.eval()
    all_latents = []

    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        _, latent = model(images)
        all_latents.append(latent.cpu())

    all_latents = torch.cat(all_latents, dim=0)  # [N, 2048]
    mu = all_latents.mean(dim=0)                  # [2048]
    return mu


# ── Global Label Map ───────────────────────────────────────────────────────
ALL_CLASSES = []
for task_labels in TASK_PARTITION.values():
    ALL_CLASSES.extend(task_labels)
GLOBAL_LABEL_MAP = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}


# ── Main Training Pipeline ─────────────────────────────────────────────────
def run_cil_training(
    batch_size:  int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> dict:

    print("\n" + "="*65)
    print("  Phase 2: ResNet-50 CIL Training — NIH Chest X-ray 14")
    print("="*65)
    print(f"  Device:      {DEVICE}")
    print(f"  Epochs/task: {NUM_EPOCHS_PER_TASK}")
    print(f"  LR:          {LR}")
    print(f"  ER buffer:   {ER_BUFFER_SIZE_PER_CLASS} samples/class")
    print(f"  Focal γ:     {FOCAL_GAMMA}")
    print("="*65 + "\n")

    # ── 1. Build DataLoaders ─────────────────────────────────────────────
    task_data = build_cil_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        verify_images=False,
    )

    # ── 2. Initialise model, ER buffer, results storage ──────────────────
    # Start with Task 0 classes; we will replace the head for each task
    model = DualOutputResNet50(
        num_classes=len(TASK_PARTITION[0])
    ).to(DEVICE)

    er_buffer  = ExperienceReplayBuffer(max_per_class=ER_BUFFER_SIZE_PER_CLASS)
    prototypes = {}   # task_id → μ(t) Tensor [2048]
    all_results = {}  # task_id → training metrics

    # ── 3. Sequential CIL Training ───────────────────────────────────────
    for task_id in range(4):
        stats        = task_data[task_id]["stats"]
        task_classes = TASK_PARTITION[task_id]
        n_classes    = stats["n_classes"]
        pos_weights  = stats["pos_weights"]

        print(f"\n{'─'*65}")
        print(f"  TASK {task_id}: {task_classes}")
        print(f"  Classes: {n_classes} | Train: {stats['train_size']:,} | "
              f"Cal: {stats['cal_size']:,}")
        print(f"{'─'*65}")

        # Replace classifier head for new task (backbone weights preserved)
        model.classifier = nn.Linear(2048, n_classes).to(DEVICE)

        # Criterion with Focal Loss + pos_weight
        criterion = FocalLoss(
            gamma=FOCAL_GAMMA,
            pos_weight=pos_weights,
        )

        # Optimiser: lower LR on backbone, higher on fresh classifier head
        optimizer = torch.optim.AdamW([
            {"params": model.features.parameters(),    "lr": LR * 0.1},
            {"params": model.classifier.parameters(),  "lr": LR},
            {"params": model.dropout.parameters(),     "lr": LR},
        ], weight_decay=WEIGHT_DECAY)

        # LR scheduler: cosine annealing over NUM_EPOCHS_PER_TASK
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS_PER_TASK, eta_min=LR * 0.01
        )

        # Get replay loader (None for Task 0)
        replay_loader = er_buffer.get_replay_loader(
            task_classes=task_classes,
            batch_size=batch_size,
            global_label_map=GLOBAL_LABEL_MAP,
        )
        if replay_loader:
            print(f"  [ER] Replaying {er_buffer.size} buffered samples.")
        else:
            print(f"  [ER] No replay buffer yet (Task 0).")

        # ── Per-task training loop ────────────────────────────────────────
        best_val_auc   = 0.0
        best_state     = None
        task_train_log = []

        for epoch in range(1, NUM_EPOCHS_PER_TASK + 1):
            t_start = time.time()

            train_metrics = train_one_epoch(
                model, task_data[task_id]["train"],
                replay_loader, optimizer, criterion,
                DEVICE, epoch, task_id,
            )
            val_metrics = evaluate(
                model, task_data[task_id]["val"],
                criterion, DEVICE, task_classes,
            )
            scheduler.step()

            elapsed = time.time() - t_start
            log_entry = {
                "epoch":     epoch,
                "train_loss": round(train_metrics["loss"], 4),
                "val_loss":   round(val_metrics["loss"], 4),
                "val_auc":    round(val_metrics["mean_auc"], 4),
                "elapsed_s":  round(elapsed, 1),
            }
            task_train_log.append(log_entry)

            # AUC per class (compact display)
            auc_str = "  ".join(
                f"{cls[:6]}={auc:.3f}"
                for cls, auc in val_metrics["per_class_auc"].items()
                if not np.isnan(auc)
            )
            print(f"  Ep {epoch:02d}/{NUM_EPOCHS_PER_TASK} | "
                  f"TrLoss={train_metrics['loss']:.4f} | "
                  f"ValLoss={val_metrics['loss']:.4f} | "
                  f"ValAUC={val_metrics['mean_auc']:.4f} | "
                  f"{elapsed:.0f}s")
            print(f"         {auc_str}")

            # Save best checkpoint for this task
            if val_metrics["mean_auc"] > best_val_auc:
                best_val_auc = val_metrics["mean_auc"]
                best_state   = {
                    "backbone":    deepcopy(model.features.state_dict()),
                    "classifier":  deepcopy(model.classifier.state_dict()),
                    "task_id":     task_id,
                    "epoch":       epoch,
                    "val_auc":     best_val_auc,
                    "task_classes": task_classes,
                }

        # ── Post-task operations ──────────────────────────────────────────
        # 1. Restore best weights for this task
        model.features.load_state_dict(best_state["backbone"])
        model.classifier.load_state_dict(best_state["classifier"])

        # 2. Evaluate on test set
        test_metrics = evaluate(
            model, task_data[task_id]["test"],
            criterion, DEVICE, task_classes,
        )
        print(f"\n  [Task {task_id} FINAL] "
              f"Best Val AUC={best_val_auc:.4f} | "
              f"Test AUC={test_metrics['mean_auc']:.4f}")
        for cls, auc in test_metrics["per_class_auc"].items():
            if not np.isnan(auc):
                print(f"    {cls:<22}: {auc:.4f}")

        # 3. Compute and save global prototype μ(t)
        print(f"\n  [Prototype] Computing μ(t={task_id}) over calibration set...")
        mu_t = compute_task_prototype(
            model, task_data[task_id]["calibration"], DEVICE
        )
        prototypes[task_id] = mu_t
        print(f"  μ(t={task_id}) shape: {mu_t.shape}  "
              f"  norm: {mu_t.norm().item():.4f}")

        # 4. Save checkpoint
        ckpt_path = CHECKPOINT_DIR / f"task{task_id}_best.pt"
        torch.save({
            "backbone":     best_state["backbone"],
            "classifier":   best_state["classifier"],
            "task_id":      task_id,
            "val_auc":      best_val_auc,
            "test_auc":     test_metrics["mean_auc"],
            "task_classes": task_classes,
            "prototype":    mu_t,
            "train_log":    task_train_log,
        }, ckpt_path)
        print(f"  [Saved] Checkpoint → {ckpt_path}")

        # 5. Fill ER buffer AFTER training this task
        print(f"\n  [ER] Filling buffer from Task {task_id} calibration set...")
        er_buffer.add_task_data(
            loader=task_data[task_id]["calibration"],
            task_id=task_id,
            task_classes=task_classes,
            device=DEVICE,
            global_label_map=GLOBAL_LABEL_MAP,
        )

        # 6. Store results
        all_results[task_id] = {
            "best_val_auc":      best_val_auc,
            "test_auc":          test_metrics["mean_auc"],
            "test_per_class_auc": test_metrics["per_class_auc"],
            "train_log":         task_train_log,
        }

    # ── 4. Compute and log backbone drift between tasks ───────────────────
    print("\n" + "="*65)
    print("  Backbone Drift Summary (L2 distance between prototypes)")
    print("="*65)
    drift_log = {}
    for t in range(1, 4):
        delta = torch.norm(prototypes[t] - prototypes[t-1]).item()
        drift_log[f"T{t-1}→T{t}"] = round(delta, 4)
        print(f"  Δμ(T{t-1}→T{t}) = {delta:.4f} L2 units")

    # ── 5. Save all prototypes and drift log ──────────────────────────────
    proto_path = CHECKPOINT_DIR / "prototypes.pt"
    torch.save({
        "prototypes": {t: mu for t, mu in prototypes.items()},
        "drift_log":  drift_log,
    }, proto_path)
    print(f"\n  [Saved] Prototypes → {proto_path}")

    # ── 6. Save results JSON ──────────────────────────────────────────────
    results_path = RESULTS_DIR / "phase2_results.json"
    serialisable = {}
    for tid, res in all_results.items():
        serialisable[str(tid)] = {
            "best_val_auc":      round(res["best_val_auc"], 4),
            "test_auc":          round(res["test_auc"], 4),
            "test_per_class_auc": {
                k: round(v, 4) if not np.isnan(v) else None
                for k, v in res["test_per_class_auc"].items()
            },
        }
    serialisable["drift_log"] = drift_log
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  [Saved] Results → {results_path}")

    print("\n[Phase 2 COMPLETE] — Backbone training finished.\n")
    return all_results, prototypes, drift_log


# ── Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    all_results, prototypes, drift_log = run_cil_training(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("\n[Final AUC Summary]")
    for task_id, res in all_results.items():
        print(f"  Task {task_id}: Val AUC={res['best_val_auc']:.4f}  "
              f"Test AUC={res['test_auc']:.4f}")
    print("\n[Backbone Drift]")
    for key, val in drift_log.items():
        print(f"  {key}: {val:.4f} L2 units")