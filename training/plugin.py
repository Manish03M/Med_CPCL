# training/plugin.py
# Med-CPCL Plugin -- correction_strength rescaled for L2 drift magnitudes

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader
from config import DEVICE, GAMMA
from conformal.conformal import score_aps
from conformal.scoring import ScoreMemory


class MedCPCLPlugin:

    def __init__(self, score_memory: ScoreMemory,
                 gamma: float = GAMMA,
                 drift_correction: float = 0.005):
        # drift_correction=0.005: correction = 0.005 * (drift/ref_drift)
        # ref_drift ~= 5.0 (expected L2 norm for 512-dim space)
        # so actual score inflation ~= 0.005 * (6.8/5.0) ~= 0.007 per score
        self.score_memory     = score_memory
        self.gamma            = gamma
        self.drift_correction = drift_correction
        self.ref_drift        = 5.0   # reference normalisation constant

    def after_training_task(self, model, cal_loader: DataLoader,
                             task_id: int, device: str = DEVICE):
        model.eval()
        all_scores, all_latents, all_labels = [], [], []

        with torch.no_grad():
            for x, y in cal_loader:
                x = x.to(device)
                logits, latents = model(x)
                probs   = torch.softmax(logits, dim=1).cpu()
                labels  = y.squeeze().long().cpu()
                scores  = score_aps(probs, labels)
                all_scores.append(scores)
                all_latents.append(latents.cpu())
                all_labels.append(labels)

        all_scores  = torch.cat(all_scores)
        all_latents = torch.cat(all_latents)
        all_labels  = torch.cat(all_labels)

        if task_id > 0:
            drift = self.score_memory.compute_backbone_drift(
                task_id, all_latents)
            # Normalise drift by reference magnitude
            normalised = drift / self.ref_drift
            effective_strength = self.drift_correction * normalised
            print(f"  [Plugin] Task {task_id} backbone drift: {drift:.4f} "
                  f"(normalised: {normalised:.3f})")

            if drift > 0.1:
                self.score_memory.apply_drift_correction(
                    drift_magnitude=normalised,
                    correction_strength=self.drift_correction)
                print(f"  [Plugin] Drift correction applied: "
                      f"+{effective_strength:.5f} per score")
            else:
                print(f"  [Plugin] Drift negligible -- no correction")
        else:
            print(f"  [Plugin] Task 0: initialising score memory")

        self.score_memory.add(all_scores, all_latents, all_labels, task_id)
        self.score_memory.update_global_prototype(task_id, all_latents)
        self.score_memory.summary()
