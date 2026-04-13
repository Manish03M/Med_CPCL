# conformal/weighted_cp.py
# WeightedCP: The core Med-CPCL contribution
#
# Implements:
#   1. Non-exchangeable weighted quantile estimator
#      q_hat = Quantile(1-alpha; sum wi*delta_si + delta_inf)
#   2. Mondrian CP: per-class separate quantiles
#   3. Full predict_set(x) API

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import ALPHA, GAMMA, DEVICE, NUM_CLASSES
from conformal.scoring import ScoreMemory
from conformal.conformal import score_aps


def weighted_quantile(scores: np.ndarray, weights: np.ndarray,
                      level: float) -> float:
    """
    Weighted quantile estimator.

    Implements the non-exchangeable quantile from thesis Section 4:
      q_hat = inf{q : sum_i wi * 1[si <= q] >= level}

    Args:
        scores  : (N,) non-conformity scores
        weights : (N,) normalised weights wi = gamma^(t - ti) / Z
        level   : float, target quantile level (1 - alpha)
    Returns:
        q_hat   : float
    """
    if len(scores) == 0:
        return 1.0  # conservative fallback

    # Sort by score
    order   = np.argsort(scores)
    s_sort  = scores[order]
    w_sort  = weights[order]

    # Weighted CDF
    wcdf    = np.cumsum(w_sort)

    # Find smallest score where weighted CDF >= level
    idx     = np.searchsorted(wcdf, level)
    if idx >= len(s_sort):
        return float(s_sort[-1]) + 1e-6   # add small margin if all below

    return float(s_sort[idx])


class WeightedCP:
    """
    Med-CPCL: Non-exchangeable conformal prediction with:
      - Time-aware weighting (wi = gamma^(t - ti))
      - Score memory (si, zi, ti) from ScoreMemory
      - Mondrian CP: separate q_hat per class
      - APS scoring for set efficiency

    This is the proposed method (Phase 9 contribution).
    """

    def __init__(self, score_memory: ScoreMemory,
                 alpha: float = ALPHA,
                 gamma: float = GAMMA,
                 use_mondrian: bool = True):
        self.score_memory  = score_memory
        self.alpha         = alpha
        self.gamma         = gamma
        self.use_mondrian  = use_mondrian
        self.q_hat_per_class = {}    # Mondrian: one q_hat per class
        self.q_hat_global    = None  # Marginal fallback

    def calibrate(self, current_task: int):
        """
        Compute weighted quantile(s) from score memory.

        Mondrian mode: separate q_hat per class
        Global mode  : single q_hat across all classes
        """
        if self.use_mondrian:
            self.q_hat_per_class = {}
            for c in range(NUM_CLASSES):
                scores, weights = self.score_memory.get_weighted_scores(
                    current_task, c, self.gamma)
                if len(scores) == 0:
                    self.q_hat_per_class[c] = 1.0
                    continue
                n     = len(scores)
                level = np.ceil((n + 1) * (1 - self.alpha)) / (n + 1e-8)
                level = min(level, 1.0)
                self.q_hat_per_class[c] = weighted_quantile(
                    scores, weights, level)
        else:
            # Marginal: pool all classes
            all_s, all_w = [], []
            for c in range(NUM_CLASSES):
                s, w = self.score_memory.get_weighted_scores(
                    current_task, c, self.gamma)
                if len(s) > 0:
                    all_s.append(s)
                    all_w.append(w)
            if all_s:
                all_s = np.concatenate(all_s)
                all_w = np.concatenate(all_w)
                all_w = all_w / (all_w.sum() + 1e-8)
                n     = len(all_s)
                level = np.ceil((n + 1) * (1 - self.alpha)) / (n + 1e-8)
                self.q_hat_global = weighted_quantile(all_s, all_w, min(level, 1.0))
            else:
                self.q_hat_global = 1.0

        return self.q_hat_per_class if self.use_mondrian else self.q_hat_global

    def predict_set(self, model, x: torch.Tensor, device=DEVICE):
        """
        Returns prediction sets using weighted quantile.

        For APS + Mondrian:
          For each candidate class c:
            Compute APS score if c were the true label
            Include c in set if score <= q_hat[c]
        """
        model.eval()
        with torch.no_grad():
            x      = x.to(device)
            logits, _ = model(x)
            probs  = torch.softmax(logits, dim=1).cpu()  # (B, C)

        B, C = probs.shape
        prediction_sets = []

        for i in range(B):
            p     = probs[i]
            order = torch.argsort(p, descending=True)
            cumsum = torch.cumsum(p[order], dim=0)

            pred_set = []
            for c in range(C):
                # APS score if c were the true class
                rank     = (order == c).nonzero(as_tuple=True)[0].item()
                aps_score = cumsum[rank].item()

                # Get threshold: Mondrian or global
                if self.use_mondrian:
                    q = self.q_hat_per_class.get(c, 1.0)
                else:
                    q = self.q_hat_global if self.q_hat_global else 1.0

                if aps_score <= q:
                    pred_set.append(c)

            # Ensure non-empty: always include argmax
            if not pred_set:
                pred_set = [p.argmax().item()]

            prediction_sets.append(sorted(pred_set))

        return prediction_sets

    def evaluate(self, model, test_loader, device=DEVICE):
        """Compute coverage and avg set size."""
        covered, set_sizes, total = 0, [], 0

        for x, y in test_loader:
            pred_sets = self.predict_set(model, x, device)
            y_list    = y.squeeze().long().tolist()
            if isinstance(y_list, int):
                y_list = [y_list]
            for label, pred_set in zip(y_list, pred_sets):
                covered   += int(label in pred_set)
                set_sizes.append(len(pred_set))
                total     += 1

        coverage     = covered / total
        avg_set_size = float(np.mean(set_sizes))
        return coverage, avg_set_size

    def print_quantiles(self):
        """Print per-class quantiles for inspection."""
        if self.use_mondrian:
            print("  Mondrian q_hat per class:")
            for c, q in self.q_hat_per_class.items():
                n = len(self.score_memory.scores[c])
                print(f"    Class {c}: q={q:.4f}  (n={n} scores)")
        else:
            print(f"  Global q_hat: {self.q_hat_global:.4f}")
