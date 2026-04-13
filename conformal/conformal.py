# conformal/conformal.py
# Conformal Prediction engine for Med-CPCL.
#
# This file implements TWO CP methods:
#
# 1. StandardCP (this phase - baseline):
#    - Split CP with standard quantile
#    - Assumes exchangeability (violated in CL)
#    - Will show coverage degradation on old tasks
#
# 2. WeightedCP (Phase 9 - your contribution):
#    - Non-exchangeable weighted quantile
#    - Time-aware weighting: wi = gamma^(t - ti)
#    - Drift compensation via prototype shift
#    - Mondrian CP for per-class guarantees
#    - APS scoring for efficiency

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import ALPHA, GAMMA, DEVICE, NUM_CLASSES


# ══════════════════════════════════════════════════════════════════════════
# NON-CONFORMITY SCORING
# ══════════════════════════════════════════════════════════════════════════

def score_aps(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Adaptive Prediction Sets (APS) non-conformity score.

    Thesis requirement: "Use cumulative probability-based scoring"
    APS sorts classes by descending probability and accumulates until
    the true class is included. This produces smaller prediction sets
    than simple 1 - p_y scoring.

    Args:
        probs  : (N, C) softmax probabilities
        labels : (N,)   true class indices
    Returns:
        scores : (N,)   non-conformity scores in [0, 1]
    """
    N = probs.size(0)
    scores = torch.zeros(N)

    for i in range(N):
        p    = probs[i]                              # (C,)
        y    = labels[i].item()
        # Sort classes by descending probability
        order = torch.argsort(p, descending=True)   # most likely first
        # Cumulative sum of probabilities
        cumsum = torch.cumsum(p[order], dim=0)
        # Find rank of true class
        rank   = (order == y).nonzero(as_tuple=True)[0].item()
        # Score = cumulative prob UP TO AND INCLUDING true class
        scores[i] = cumsum[rank].item()

    return scores


def score_softmax(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Standard softmax non-conformity score: s = 1 - p_y
    Simpler than APS but less efficient (larger sets).
    Used in StandardCP baseline.
    """
    N = probs.size(0)
    scores = torch.zeros(N)
    for i in range(N):
        scores[i] = 1.0 - probs[i, labels[i].item()].item()
    return scores


# ══════════════════════════════════════════════════════════════════════════
# STANDARD CONFORMAL PREDICTION (Phase 8 baseline)
# ══════════════════════════════════════════════════════════════════════════

class StandardCP:
    """
    Split Conformal Prediction with standard (unweighted) quantile.

    Calibration:
        q = (1-alpha)(1 + 1/n) quantile of {s_1, ..., s_n}

    Prediction:
        C(x) = {y : s(x,y) <= q}

    Limitation (thesis proof-of-failure):
        Assumes calibration scores are exchangeable with test scores.
        In CL, representational drift breaks this assumption -- scores
        computed on Task 0 calibration data become stale after Task 1+
        training, causing coverage to drop below 1-alpha.
    """

    def __init__(self, alpha: float = ALPHA, scoring: str = "aps"):
        self.alpha   = alpha
        self.scoring = scoring
        self.q_hat   = None         # calibrated quantile
        self.cal_scores = None      # stored for inspection

    def calibrate(self, model, cal_loader, device=DEVICE):
        """
        Compute non-conformity scores on calibration set.
        Stores q_hat = (1-alpha) quantile with finite-sample correction.
        """
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for x, y in cal_loader:
                x = x.to(device)
                logits, _ = model(x)
                probs = torch.softmax(logits, dim=1).cpu()
                all_probs.append(probs)
                all_labels.append(y.squeeze().long().cpu())

        probs  = torch.cat(all_probs)
        labels = torch.cat(all_labels)

        if self.scoring == "aps":
            scores = score_aps(probs, labels)
        else:
            scores = score_softmax(probs, labels)

        self.cal_scores = scores

        # Finite-sample corrected quantile
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_hat = float(torch.quantile(scores, level))

        return self.q_hat

    def predict(self, model, x: torch.Tensor, device=DEVICE):
        """
        Returns prediction set C(x) for a single batch.

        Args:
            x : (B, 3, 28, 28)
        Returns:
            prediction_sets : list of lists, each containing class indices
        """
        assert self.q_hat is not None, "Must calibrate before predicting"
        model.eval()

        with torch.no_grad():
            x = x.to(device)
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1).cpu()  # (B, C)

        prediction_sets = []
        for i in range(probs.size(0)):
            if self.scoring == "aps":
                # Include classes until cumulative prob exceeds q_hat
                p     = probs[i]
                order = torch.argsort(p, descending=True)
                cumsum = torch.cumsum(p[order], dim=0)
                # Include all classes where cumsum <= q_hat, plus the one that crosses
                include = (cumsum <= self.q_hat)
                # Always include at least the first class that crosses
                if not include.any():
                    include[0] = True
                else:
                    # Include the class that pushes cumsum over threshold
                    first_over = include.sum().item()
                    if first_over < len(order):
                        include[first_over] = True
                pred_set = order[include].tolist()
            else:
                pred_set = [c for c in range(probs.size(1))
                            if (1 - probs[i, c].item()) <= self.q_hat]

            prediction_sets.append(sorted(pred_set))

        return prediction_sets

    def evaluate(self, model, test_loader, device=DEVICE):
        """
        Compute marginal coverage and average set size on test_loader.

        Returns:
            coverage   : fraction of samples where true label in set
            avg_set_size: mean prediction set size
        """
        assert self.q_hat is not None, "Must calibrate before evaluating"
        model.eval()

        covered, set_sizes, total = 0, [], 0

        with torch.no_grad():
            for x, y in test_loader:
                pred_sets = self.predict(model, x, device)
                y_list    = y.squeeze().long().tolist()
                if isinstance(y_list, int):
                    y_list = [y_list]

                for label, pred_set in zip(y_list, pred_sets):
                    covered   += int(label in pred_set)
                    set_sizes.append(len(pred_set))
                    total     += 1

        coverage      = covered / total
        avg_set_size  = float(np.mean(set_sizes))
        return coverage, avg_set_size
