# experiments/run_organamnist.py
# Second dataset experiment: OrganAMNIST
#
# Generalisability test for Med-CPCL beyond BloodMNIST.
# OrganAMNIST: 11 organ classes from abdominal CT scans (grayscale).
# Split into 4 tasks: [0,1,2], [3,4,5], [6,7,8], [9,10]
#
# Key difference from BloodMNIST:
#   - 1-channel grayscale (vs 3-channel RGB)
#   - 11 classes (vs 8)
#   - CT imaging domain (vs microscopy)
#   - Model first conv adapted for 1-channel input

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, models
from medmnist import OrganAMNIST
import medmnist

from config import (DATA_DIR, BATCH_SIZE, NUM_EPOCHS, DEVICE,
                    TABLES_DIR, CKPT_DIR, SEED, LR, WEIGHT_DECAY,
                    ALPHA, GAMMA, SCORE_MEMORY_SIZE, REPLAY_BUFFER_SIZE)
from training.train import Trainer, evaluate_all_tasks, train_one_epoch
from training.replay_buffer import ReplayBuffer
from training.plugin import MedCPCLPlugin
from conformal.scoring import ScoreMemory
from conformal.weighted_cp import WeightedCP
from conformal.conformal import StandardCP
from experiments.run_finetuning import compute_metrics
from experiments.run_replay import ERTrainer

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── OrganAMNIST constants ──────────────────────────────────────────────────
ORGAN_NUM_CLASSES  = 11
ORGAN_NUM_TASKS    = 4
# Task splits: 3+3+3+2 = 11
ORGAN_TASK_MAP = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8],
    3: [9, 10],
}

ORGAN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])   # 1-channel
])


# ── Adapted model for 1-channel input ──────────────────────────────────────

class OrganModel(nn.Module):
    """ResNet-18 adapted for OrganAMNIST (1-channel, 28x28, 11 classes)."""

    def __init__(self, num_classes=ORGAN_NUM_CLASSES):
        super().__init__()
        backbone = models.resnet18(weights=None)
        # Adapt for 1-channel grayscale
        backbone.conv1  = nn.Conv2d(1, 64, kernel_size=3,
                                     stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        self.features   = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool, backbone.layer1, backbone.layer2,
            backbone.layer3, backbone.layer4, backbone.avgpool,
        )
        self.classifier = nn.Linear(512, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z      = self.features(x)
        z      = torch.flatten(z, 1)
        logits = self.classifier(z)
        return logits, z


# ── Data loading ───────────────────────────────────────────────────────────

def _get_indices(dataset, class_list):
    labels = np.array([dataset[i][1] for i in range(len(dataset))]).squeeze()
    return np.where(np.isin(labels, class_list))[0]


def get_organ_task_loaders(task_id, cal_fraction=0.2):
    class_list = ORGAN_TASK_MAP[task_id]

    train_full = OrganAMNIST(split="train", transform=ORGAN_TRANSFORM,
                              download=True, root=DATA_DIR)
    test_full  = OrganAMNIST(split="test",  transform=ORGAN_TRANSFORM,
                              download=True, root=DATA_DIR)

    train_idx  = _get_indices(train_full, class_list)
    test_idx   = _get_indices(test_full,  class_list)

    train_sub  = Subset(train_full, train_idx)
    test_sub   = Subset(test_full,  test_idx)

    n_cal   = max(1, int(len(train_sub) * cal_fraction))
    n_train = len(train_sub) - n_cal
    tr_split, cal_split = random_split(
        train_sub, [n_train, n_cal],
        generator=torch.Generator().manual_seed(SEED)
    )

    tr  = DataLoader(tr_split,  batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=2, pin_memory=True)
    cal = DataLoader(cal_split, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=2, pin_memory=True)
    te  = DataLoader(test_sub,  batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=2, pin_memory=True)
    return tr, cal, te


def inspect_organ():
    info   = medmnist.INFO["organamnist"]
    labels = list(info["label"].values())
    print("  OrganAMNIST classes:", labels)
    print("  Task map:")
    for t, cls in ORGAN_TASK_MAP.items():
        print(f"    Task {t}: {[labels[c] for c in cls]}")


# ── Run experiment ─────────────────────────────────────────────────────────

def run_organamnist():
    print("=" * 65)
    print("  ORGANAMNIST: Generalisability Experiment")
    print("  11 classes, 4 tasks, 1-channel CT images")
    print("=" * 65)
    inspect_organ()

    model         = OrganModel().to(DEVICE)
    replay_buffer = ReplayBuffer(max_per_class=REPLAY_BUFFER_SIZE)
    score_memory  = ScoreMemory(num_classes=ORGAN_NUM_CLASSES)
    plugin        = MedCPCLPlugin(score_memory)
    wcp           = WeightedCP(score_memory, alpha=ALPHA,
                               gamma=GAMMA, use_mondrian=True)
    std_cp        = StandardCP(alpha=ALPHA, scoring="aps")

    # Use ERTrainer but with organ model
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(model.parameters(),
                                   lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                                   gamma=0.5)

    acc_matrix   = np.zeros((ORGAN_NUM_TASKS, ORGAN_NUM_TASKS))
    test_loaders = []
    cp_results   = {}

    for task_id in range(ORGAN_NUM_TASKS):
        print(f"\n[Task {task_id}] classes={ORGAN_TASK_MAP[task_id]}")
        tr, cal, te = get_organ_task_loaders(task_id)
        test_loaders.append(te)

        # Collect current + replay data
        if replay_buffer.buffer:
            combined = replay_buffer.get_combined_loader(tr)
        else:
            combined = tr

        # Train
        print(f"  Training {len(tr.dataset)} samples, {NUM_EPOCHS} epochs")
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for x, y in combined:
                x = x.to(DEVICE)
                y = y.squeeze().long().to(DEVICE)
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                correct    += (logits.argmax(1) == y).sum().item()
                total      += x.size(0)
            scheduler.step()
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1:02d}/{NUM_EPOCHS}  "
                      f"loss={total_loss/total:.4f}  "
                      f"acc={correct/total:.4f}")

        replay_buffer.update(tr, task_id)
        plugin.after_training_task(model, cal, task_id)
        wcp.calibrate(current_task=task_id)
        std_cp.calibrate(model, cal)

        # Evaluate accuracy
        def eval_loader(loader):
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(DEVICE)
                    y = y.squeeze().long().to(DEVICE)
                    logits, _ = model(x)
                    correct += (logits.argmax(1) == y).sum().item()
                    total   += x.size(0)
            return correct / total

        print(f"  Accuracy on all tasks:")
        for j in range(task_id + 1):
            acc = eval_loader(test_loaders[j])
            acc_matrix[task_id][j] = acc
            print(f"    Task {j}: {acc:.4f}")

        # CP evaluation
        cp_results[task_id] = {}
        print(f"  CP (Standard vs Weighted):")
        for j in range(task_id + 1):
            std_cov, std_sz = std_cp.evaluate(model, test_loaders[j])
            wt_cov,  wt_sz  = wcp.evaluate(model,     test_loaders[j])
            cp_results[task_id][j] = {
                "standard": {"coverage": std_cov, "set_size": std_sz},
                "weighted": {"coverage": wt_cov,  "set_size": wt_sz}
            }
            print(f"    T{j}: Std={std_cov:.3f}/{std_sz:.2f}  "
                  f"Wt={wt_cov:.3f}/{wt_sz:.2f}")

    AA, BWT, FM = compute_metrics(acc_matrix)

    print("\n" + "=" * 65)
    print("  ORGANAMNIST FINAL RESULTS")
    print("=" * 65)
    print(f"  AA={AA:.4f}  BWT={BWT:.4f}  FM={FM:.4f}")

    final = cp_results[ORGAN_NUM_TASKS - 1]
    print("\n  CP after final task:")
    print(f"  {'Task':<6} {'Std_Cov':>9} {'Std_Sz':>9} "
          f"{'Wt_Cov':>9} {'Wt_Sz':>9}")
    for j in range(ORGAN_NUM_TASKS):
        s = final[j]["standard"]
        w = final[j]["weighted"]
        print(f"  T{j:<5} {s['coverage']:>9.4f} {s['set_size']:>9.4f} "
              f"{w['coverage']:>9.4f} {w['set_size']:>9.4f}")
    print("=" * 65)

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,   exist_ok=True)

    results = {
        "method"     : "med_cpcl_organamnist",
        "dataset"    : "organamnist",
        "timestamp"  : datetime.now().isoformat(),
        "AA"         : float(AA),
        "BWT"        : float(BWT),
        "FM"         : float(FM),
        "acc_matrix" : acc_matrix.tolist(),
        "cp_results" : {
            str(ti): {str(tj): v for tj, v in inner.items()}
            for ti, inner in cp_results.items()
        }
    }

    path = os.path.join(TABLES_DIR, "organamnist_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {path}")

    ckpt = os.path.join(CKPT_DIR, "organamnist_final.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"  Checkpoint: {ckpt}")

    return results


if __name__ == "__main__":
    run_organamnist()
