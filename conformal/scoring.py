# conformal/scoring.py
# Score Memory and drift compensation for Med-CPCL.
# FIXED: drift now uses replay buffer features for cross-task backbone drift

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import GAMMA, SCORE_MEMORY_SIZE, DEVICE, NUM_CLASSES


class ScoreMemory:
    """
    Stores calibration non-conformity scores with latent features and task IDs.
    Each entry: (si, zi, ti, class_label)
    """

    def __init__(self, max_per_class: int = SCORE_MEMORY_SIZE,
                 num_classes: int = NUM_CLASSES):
        self.max_per_class  = max_per_class
        self.num_classes    = num_classes
        self.scores  = {c: [] for c in range(num_classes)}
        self.latents = {c: [] for c in range(num_classes)}
        self.tasks   = {c: [] for c in range(num_classes)}
        # Global backbone snapshots: {task_id: mean_latent_vector (512,)}
        self.global_prototypes = {}

    def add(self, scores: torch.Tensor, latents: torch.Tensor,
            labels: torch.Tensor, task_id: int):
        scores  = scores.cpu().numpy()
        latents = latents.cpu().numpy()
        labels  = labels.cpu().numpy().astype(int)

        for i in range(len(scores)):
            c = int(labels[i])
            if c not in self.scores:
                continue
            self.scores[c].append(float(scores[i]))
            self.latents[c].append(latents[i])
            self.tasks[c].append(task_id)

            if len(self.scores[c]) > self.max_per_class:
                self.scores[c].pop(0)
                self.latents[c].pop(0)
                self.tasks[c].pop(0)

    def update_global_prototype(self, task_id: int, latents: torch.Tensor):
        """
        Store mean latent vector across ALL samples in cal set for this task.
        Used to measure backbone drift between tasks regardless of class overlap.
        """
        self.global_prototypes[task_id] = latents.cpu().mean(dim=0)

    def compute_backbone_drift(self, task_id: int,
                                new_latents: torch.Tensor) -> float:
        """
        Estimate global backbone drift between task t-1 and task t.
        Uses ALL latent features (not per-class) to handle disjoint class tasks.

        delta = || mu(t) - mu(t-1) ||_2

        Returns drift magnitude (scalar).
        """
        if task_id - 1 not in self.global_prototypes:
            return 0.0

        new_proto = new_latents.cpu().mean(dim=0)
        old_proto = self.global_prototypes[task_id - 1]
        drift     = float((new_proto - old_proto).norm().item())
        return drift

    def apply_drift_correction(self, drift_magnitude: float,
                                correction_strength: float = 0.05):
        """
        Apply drift correction to ALL stored old scores.
        Inflate old scores proportionally to drift magnitude and score age.

        Higher drift -> more correction needed for old calibration scores.
        """
        for c in range(self.num_classes):
            for i in range(len(self.scores[c])):
                age        = 1
                correction = correction_strength * drift_magnitude * age
                self.scores[c][i] = min(1.0, self.scores[c][i] + correction)

    def get_weighted_scores(self, current_task: int, class_label: int,
                            gamma: float = GAMMA):
        """
        Retrieve time-weighted scores for a specific class.
        wi = gamma^(current_task - ti)
        """
        if not self.scores[class_label]:
            return np.array([]), np.array([])

        scores   = np.array(self.scores[class_label])
        task_ids = np.array(self.tasks[class_label])
        weights  = gamma ** (current_task - task_ids)
        weights  = weights / (weights.sum() + 1e-8)
        return scores, weights

    def summary(self):
        total = sum(len(v) for v in self.scores.values())
        print(f"  ScoreMemory: {total} total scores across "
              f"{sum(1 for v in self.scores.values() if v)} classes")
        for c in range(self.num_classes):
            if self.scores[c]:
                tasks_seen = sorted(set(self.tasks[c]))
                print(f"    Class {c}: {len(self.scores[c])} scores | "
                      f"tasks seen: {tasks_seen}")
