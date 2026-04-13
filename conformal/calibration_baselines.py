# conformal/calibration_baselines.py
# Calibration Baselines: Temperature Scaling + Platt Scaling
#
# Thesis role (Section 6 - Baselines):
#   - Proves post-hoc calibration methods fail under representational drift
#   - Temperature Scaling: single scalar T applied to all logits
#   - Platt Scaling: per-class affine transform (W, b) on logits
#   - Both are heuristic -- NO coverage guarantees (unlike CP)
#   - ECE comparison shows they appear calibrated but fail on old tasks

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from config import DEVICE, NUM_CLASSES, CKPT_DIR, TABLES_DIR
import json


# ── Expected Calibration Error ─────────────────────────────────────────────

def compute_ece(probs, labels, n_bins=15):
    """
    Expected Calibration Error (ECE).
    Bins predictions by confidence, measures |acc - conf| per bin.

    Args:
        probs  : (N, C) softmax probabilities
        labels : (N,)   true class indices
        n_bins : number of confidence bins
    Returns:
        ece    : scalar float
        bin_data: list of (confidence, accuracy, count) per bin
    """
    confidences, predictions = probs.max(dim=1)
    accuracies  = predictions.eq(labels)

    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece       = torch.tensor(0.0)
    bin_data  = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_data.append((((lo + hi) / 2).item(), 0.0, 0))
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = accuracies[mask].float().mean()
        bin_size = mask.sum()
        ece     += (bin_size / len(labels)) * (bin_acc - bin_conf).abs()
        bin_data.append((bin_conf.item(), bin_acc.item(), bin_size.item()))

    return ece.item(), bin_data


def collect_logits_labels(model, loader, device=DEVICE):
    """Collect all logits and labels from a loader. Returns (logits, labels)."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _ = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y.squeeze().long().cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


# ── Temperature Scaling ────────────────────────────────────────────────────

class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration: divide logits by learned scalar T.
    T > 1 softens predictions (reduces overconfidence).
    T < 1 sharpens predictions.

    Optimised via NLL on calibration set.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.05)

    def fit(self, logits, labels, lr=0.01, max_iter=100):
        """Fit temperature on calibration logits/labels."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr,
                                       max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = self(logits)
            loss   = criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()


# ── Platt Scaling ──────────────────────────────────────────────────────────

class PlattScaling(nn.Module):
    """
    Per-class affine calibration: W * logits + b
    More expressive than temperature scaling.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.W = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits):
        return self.W * logits + self.b

    def fit(self, logits, labels, lr=0.01, max_iter=200):
        optimizer = torch.optim.LBFGS([self.W, self.b], lr=lr,
                                       max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = self(logits)
            loss   = criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)


# ── Main Evaluation ────────────────────────────────────────────────────────

def evaluate_calibration(model, cal_loaders, test_loaders, method_name):
    """
    For each task:
      1. Fit calibrator on cal_loader
      2. Evaluate ECE before/after calibration on test_loader
      3. Report coverage (used as baseline for CP comparison)
    """
    results = {}

    for task_id, (cal_loader, test_loader) in enumerate(
            zip(cal_loaders, test_loaders)):

        # Collect logits
        cal_logits,  cal_labels  = collect_logits_labels(model, cal_loader)
        test_logits, test_labels = collect_logits_labels(model, test_loader)

        # ECE before calibration
        raw_probs          = torch.softmax(test_logits, dim=1)
        ece_before, _      = compute_ece(raw_probs, test_labels)

        # Fit calibrator
        if method_name == "temperature":
            calibrator = TemperatureScaling()
            T = calibrator.fit(cal_logits, cal_labels)
            label_extra = f"T={T:.3f}"
        else:
            calibrator = PlattScaling()
            calibrator.fit(cal_logits, cal_labels)
            label_extra = "fitted"

        # ECE after calibration
        with torch.no_grad():
            cal_out     = calibrator(test_logits)
            cal_probs   = torch.softmax(cal_out, dim=1)
        ece_after, _    = compute_ece(cal_probs, test_labels)

        # Accuracy (calibration does not change predictions)
        preds    = raw_probs.argmax(dim=1)
        accuracy = (preds == test_labels).float().mean().item()

        results[task_id] = {
            "accuracy"  : accuracy,
            "ece_before": ece_before,
            "ece_after" : ece_after,
            "extra"     : label_extra
        }

        print(f"  Task {task_id}: acc={accuracy:.4f}  "
              f"ECE {ece_before:.4f} -> {ece_after:.4f}  [{label_extra}]")

    return results


def run_calibration_baselines(model):
    """
    Runs both Temperature Scaling and Platt Scaling on the ER-trained model.
    Prints ECE comparison per task. Returns full results dict.
    """
    from data.data_loader import get_task_loaders

    print("=" * 60)
    print("  PHASE 7: Calibration Baselines")
    print("  Model: ER-trained (replay_final.pt)")
    print("=" * 60)

    cal_loaders, test_loaders = [], []
    for t in range(4):
        _, cal_l, test_l = get_task_loaders(t)
        cal_loaders.append(cal_l)
        test_loaders.append(test_l)

    all_results = {}

    for method in ["temperature", "platt"]:
        print(f"\n-- {method.upper()} SCALING --")
        res = evaluate_calibration(model, cal_loaders, test_loaders, method)
        all_results[method] = res

    # Summary table
    print("\n" + "=" * 60)
    print("  SUMMARY: ECE Before vs After Calibration")
    print(f"  {'Task':<6} {'Acc':>6} {'ECE_raw':>9} {'ECE_TS':>9} {'ECE_PS':>9}")
    print("-" * 60)
    for t in range(4):
        acc      = all_results["temperature"][t]["accuracy"]
        ece_raw  = all_results["temperature"][t]["ece_before"]
        ece_ts   = all_results["temperature"][t]["ece_after"]
        ece_ps   = all_results["platt"][t]["ece_after"]
        print(f"  T{t:<5} {acc:>6.4f} {ece_raw:>9.4f} {ece_ts:>9.4f} {ece_ps:>9.4f}")
    print("=" * 60)
    print("  NOTE: Low ECE looks good but provides NO coverage guarantees.")
    print("  A model can be ECE-calibrated yet miss 30% of cases on Task 0.")
    print("  This motivates Conformal Prediction (Phase 8).")

    os.makedirs(TABLES_DIR, exist_ok=True)
    out_path = os.path.join(TABLES_DIR, "calibration_baselines.json")

    # Convert to serialisable format
    serial = {}
    for method, res in all_results.items():
        serial[method] = {str(k): v for k, v in res.items()}
    with open(out_path, "w") as f:
        json.dump(serial, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return all_results


if __name__ == "__main__":
    from models.model import build_model
    model = build_model()
    ckpt  = os.path.join(CKPT_DIR, "replay_final.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"  Loaded ER checkpoint: {ckpt}")
    run_calibration_baselines(model)
