"""
Phase 3: Dynamic γ Controller — Drift-Adaptive Decay Mechanism
Med-CPCL Extended Framework — NIH Chest X-ray 14

Key fix vs. original Med-CPCL:
  Latents are L2-normalised → all vectors on unit hypersphere.
  ref_drift is calibrated to unit-sphere geometry (max L2 = 2.0),
  NOT to raw 512-dim or 2048-dim Gaussian expectations.
"""

import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

# Import prior phases
from phase1_data_pipeline import build_cil_dataloaders, TASK_PARTITION, SEED
from phase2_backbone_training import (
    DualOutputResNet50,
    DEVICE,
    CHECKPOINT_DIR as PHASE2_CKPT_DIR,
    GLOBAL_LABEL_MAP,
)

torch.manual_seed(SEED)

# ── Output Paths ───────────────────────────────────────────────────────────
PHASE3_DIR = Path("./results/phase3")
PHASE3_DIR.mkdir(parents=True, exist_ok=True)

# ── Controller Hyperparameters ─────────────────────────────────────────────
# Unit-sphere calibration:
#   Your drift values: 0.13–0.19 L2 units on unit sphere
#   Maximum possible: 2.0 (antipodal vectors)
#   We set ref_drift = 0.20 so Δ̂ = Δμ / ref_drift ≈ 0.65–0.96
#   → γ ≈ exp(-1.0 × 0.65) ≈ 0.52  [high-drift transition]
#   → γ ≈ exp(-1.0 × 0.60) ≈ 0.55
#   This gives meaningful decay without collapsing to zero.
REF_DRIFT   = 0.20   # Calibrated to unit-sphere geometry
GAMMA_MIN   = 0.70   # Floor: prevents quantile collapse with too few samples
ETA         = 1.0    # Sensitivity factor — start with 1.0
ALPHA       = 0.10   # Significance level → 90% coverage

# Score Memory config
SCORE_MEMORY_SIZE_PER_CLASS = 200   # 200 entries per class
CORRECTION_STRENGTH = 0.005        # λ for score inflation


# ── Dynamic γ Controller ───────────────────────────────────────────────────
class DynamicGammaController:
    """
    Computes the adaptive decay factor γₜ as a function of backbone drift.

    Formula:
        γₜ = max(γ_min, exp(−η · Δ̂ₜ))
        where Δ̂ₜ = Δμ / ref_drift  (normalised drift)

    Product-weight accumulation:
        wᵢ(t) = ∏ₖ₌ₜᵢ₊₁ᵗ γₖ

    This means a score stored during task tᵢ gets down-weighted by the
    product of all γ values that occurred during tasks after tᵢ.
    A high-drift transition (large Δμ) produces a small γ, aggressively
    discounting stale calibration evidence from before that transition.
    """

    def __init__(
        self,
        ref_drift:  float = REF_DRIFT,
        gamma_min:  float = GAMMA_MIN,
        eta:        float = ETA,
    ):
        self.ref_drift = ref_drift
        self.gamma_min = gamma_min
        self.eta       = eta

        self.gamma_history: list[float] = []   # γₜ for each task t ≥ 1
        self.drift_history: list[float] = []   # Δμ for each transition

    def update(self, delta_mu: float) -> float:
        """
        Compute and store γₜ for the current task transition.

        Args:
            delta_mu: L2 distance ‖μ(t) − μ(t−1)‖₂

        Returns:
            gamma_t: the computed decay factor for this transition
        """
        delta_hat = delta_mu / self.ref_drift
        gamma_t   = max(self.gamma_min, math.exp(-self.eta * delta_hat))

        self.drift_history.append(round(delta_mu, 6))
        self.gamma_history.append(round(gamma_t, 6))

        return gamma_t

    def get_weight(self, stored_task: int, current_task: int) -> float:
        """
        Compute the accumulated product weight for a score stored during
        `stored_task` when the current task is `current_task`.

        w(stored_task, current_task) = ∏ₖ₌stored_task+1ᵗ γₖ

        For stored_task == current_task: weight = 1.0 (no decay)
        For stored_task == 0, current_task == 3: weight = γ₁ × γ₂ × γ₃

        Args:
            stored_task:   task index when the score was computed
            current_task:  current task index

        Returns:
            weight: float in [γ_min^(current-stored), 1.0]
        """
        if stored_task >= current_task:
            return 1.0

        # γ_history[k] corresponds to the transition from task k to task k+1
        # i.e., gamma_history[0] = γ₁ (T0→T1 transition)
        weight = 1.0
        for k in range(stored_task, current_task):
            if k < len(self.gamma_history):
                weight *= self.gamma_history[k]
            else:
                weight *= self.gamma_min  # fallback
        return weight

    def summary(self) -> dict:
        return {
            "ref_drift":     self.ref_drift,
            "gamma_min":     self.gamma_min,
            "eta":           self.eta,
            "drift_history": self.drift_history,
            "gamma_history": self.gamma_history,
        }


# ── APS Non-Conformity Score for Multi-Label ──────────────────────────────
def compute_aps_scores_multilabel(
    logits:      torch.Tensor,
    labels:      torch.Tensor,
    task_classes: list[str],
) -> dict[str, torch.Tensor]:
    """
    Computes per-class APS non-conformity scores for multi-label classification.

    In the multi-label case, each class is treated as an independent binary
    classifier. The APS score for class c on sample i is:

        s(xᵢ, c) = 1 − σ(logit_c(xᵢ))

    This equals the probability the model assigns to the NEGATIVE class.
    A low score means the model is confident the class IS present.
    A high score means the model is uncertain or predicts absence.

    At calibration time we only store s for samples where label_c = 1
    (true positives), so the quantile q̂_c represents the threshold below
    which 90% of true-positive samples fall.

    Args:
        logits:      [B, n_classes] raw logits
        labels:      [B, n_classes] binary multi-hot targets
        task_classes: list of class names for this task

    Returns:
        scores_dict: {class_name: Tensor of APS scores for positive samples}
    """
    probs  = torch.sigmoid(logits)           # [B, n_classes]
    scores = 1.0 - probs                     # [B, n_classes] — nonconformity

    scores_dict = {}
    for i, cls_name in enumerate(task_classes):
        pos_mask = labels[:, i] == 1.0      # only keep true positives
        if pos_mask.sum() > 0:
            scores_dict[cls_name] = scores[pos_mask, i].cpu()
        else:
            scores_dict[cls_name] = torch.tensor([])

    return scores_dict


# ── Score Memory ──────────────────────────────────────────────────────────
@dataclass
class ScoreMemoryEntry:
    """Single entry in the Score Memory."""
    score:    float   # sᵢ — APS non-conformity score
    latent:   torch.Tensor  # zᵢ — L2-normalised [2048] vector
    task_id:  int     # tᵢ — task when this sample was calibrated
    cls_name: str     # which class this score belongs to


class ScoreMemory:
    """
    Stores (sᵢ, zᵢ, tᵢ) triplets per class with FIFO eviction.

    FIFO intentionally biases toward recent calibration evidence —
    the same design choice as the original Med-CPCL Score Memory.
    """

    def __init__(self, max_per_class: int = SCORE_MEMORY_SIZE_PER_CLASS):
        self.max_per_class = max_per_class
        # cls_name → list of ScoreMemoryEntry (FIFO order, newest last)
        self.memory: dict[str, list[ScoreMemoryEntry]] = defaultdict(list)

    def add(
        self,
        scores_dict: dict[str, torch.Tensor],
        latents:     torch.Tensor,
        labels:      torch.Tensor,
        task_classes: list[str],
        task_id:     int,
    ) -> None:
        """
        Adds calibration samples to the Score Memory.

        Args:
            scores_dict:  {cls_name: [K] APS scores for positives of that class}
            latents:      [B, 2048] latent vectors for the FULL batch
            labels:       [B, n_classes] binary labels for the batch
            task_classes: list of class names for this task
            task_id:      current task index
        """
        for cls_idx, cls_name in enumerate(task_classes):
            if cls_name not in scores_dict or len(scores_dict[cls_name]) == 0:
                continue

            # Get latents of positive samples for this class
            pos_mask = labels[:, cls_idx] == 1.0
            pos_latents = latents[pos_mask].cpu()  # [K, 2048]
            cls_scores  = scores_dict[cls_name]    # [K]

            for k in range(min(len(cls_scores), len(pos_latents))):
                entry = ScoreMemoryEntry(
                    score=float(cls_scores[k].item()),
                    latent=pos_latents[k].clone(),
                    task_id=task_id,
                    cls_name=cls_name,
                )
                self.memory[cls_name].append(entry)

            # FIFO eviction: keep only the most recent max_per_class entries
            if len(self.memory[cls_name]) > self.max_per_class:
                self.memory[cls_name] = self.memory[cls_name][-self.max_per_class:]

    def apply_drift_correction(
        self,
        delta_mu: float,
        correction_strength: float = CORRECTION_STRENGTH,
        ref_drift: float = REF_DRIFT,
    ) -> None:
        """
        Inflates ALL stored scores proportional to measured backbone drift.
        Makes old scores more conservative to prevent undercoverage.

        sᵢ ← min(1.0, sᵢ + λ · (Δμ / ref_drift))

        Args:
            delta_mu:            current task's drift magnitude
            correction_strength: λ hyperparameter
            ref_drift:           normalisation constant
        """
        correction = correction_strength * (delta_mu / ref_drift)
        n_corrected = 0
        for cls_entries in self.memory.values():
            for entry in cls_entries:
                entry.score = min(1.0, entry.score + correction)
                n_corrected += 1
        return n_corrected, correction

    def get_scores_and_weights(
        self,
        cls_name:    str,
        current_task: int,
        controller:  DynamicGammaController,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns scores and normalised product weights for a given class.

        Args:
            cls_name:     class to retrieve
            current_task: current task index for weight computation
            controller:   DynamicGammaController instance

        Returns:
            scores:  [N] numpy array of APS scores
            weights: [N] numpy array (sum-to-one product weights)
        """
        entries = self.memory.get(cls_name, [])
        if not entries:
            return np.array([]), np.array([])

        scores  = np.array([e.score for e in entries])
        raw_w   = np.array([
            controller.get_weight(e.task_id, current_task)
            for e in entries
        ])

        # Normalise weights to sum to 1
        w_sum = raw_w.sum()
        if w_sum > 0:
            weights = raw_w / w_sum
        else:
            weights = np.ones(len(raw_w)) / len(raw_w)

        return scores, weights

    @property
    def total_entries(self) -> int:
        return sum(len(v) for v in self.memory.values())

    def class_summary(self) -> dict:
        return {cls: len(entries) for cls, entries in self.memory.items()}


# ── Weighted Quantile Estimator ────────────────────────────────────────────
def weighted_quantile(
    scores:  np.ndarray,
    weights: np.ndarray,
    level:   float,
) -> float:
    """
    Computes the weighted quantile at coverage level (1 − α).

    Uses the Weighted Empirical CDF formulation from Med-CPCL:
        q̂ = inf{q : Σᵢ w̃ᵢ · 𝟙[sᵢ ≤ q] ≥ ⌈(n+1)(1−α)⌉/n}

    With the product-weight variant from the implementation plan:
        weights reflect accumulated γ decay across task transitions.

    Args:
        scores:  [N] APS scores for a given class
        weights: [N] normalised product weights (sum to 1)
        level:   1 − α = 0.90 for 90% coverage

    Returns:
        q_hat: conformal quantile threshold
    """
    if len(scores) == 0:
        return 1.0  # degenerate: no calibration data → include all

    n = len(scores)
    # Finite-sample correction
    corrected_level = math.ceil((n + 1) * level) / n
    corrected_level = min(corrected_level, 1.0)

    # Sort by score
    order   = np.argsort(scores)
    s_sort  = scores[order]
    w_sort  = weights[order]

    # Weighted ECDF: cumulative weight up to each score
    cum_w = np.cumsum(w_sort)

    # Find smallest s such that cum_w ≥ corrected_level
    idx = np.searchsorted(cum_w, corrected_level)
    if idx >= len(s_sort):
        return 1.0  # all weights exhausted before reaching level
    return float(s_sort[idx])


# ── Model Loading ──────────────────────────────────────────────────────────
def load_task_model(task_id: int) -> DualOutputResNet50:
    """Loads the best checkpoint for a given task."""
    ckpt_path = PHASE2_CKPT_DIR / f"task{task_id}_best.pt"
    ckpt      = torch.load(ckpt_path, map_location=DEVICE)

    n_classes = len(TASK_PARTITION[task_id])
    model     = DualOutputResNet50(num_classes=n_classes).to(DEVICE)
    model.features.load_state_dict(ckpt["backbone"])
    model.classifier.load_state_dict(ckpt["classifier"])
    model.eval()
    return model


# ── Main Phase 3 Pipeline ─────────────────────────────────────────────────
def run_phase3(batch_size: int = 32, num_workers: int = 4) -> dict:

    print("\n" + "="*65)
    print("  Phase 3: Dynamic γ Controller + Score Memory Construction")
    print("="*65)
    print(f"  ref_drift:           {REF_DRIFT}  (unit-sphere calibrated)")
    print(f"  gamma_min:           {GAMMA_MIN}")
    print(f"  eta:                 {ETA}")
    print(f"  correction_strength: {CORRECTION_STRENGTH}")
    print(f"  alpha:               {ALPHA}  (→ {int((1-ALPHA)*100)}% coverage)")
    print(f"  score_memory_size:   {SCORE_MEMORY_SIZE_PER_CLASS}/class")
    print("="*65 + "\n")

    # ── 1. Load prototypes from Phase 2 ──────────────────────────────────
    print("[1/5] Loading Phase 2 prototypes and drift log...")
    proto_data  = torch.load(PHASE2_CKPT_DIR / "prototypes.pt", map_location="cpu")
    prototypes  = proto_data["prototypes"]   # {task_id: Tensor[2048]}
    p2_drifts   = proto_data["drift_log"]    # {"T0→T1": 0.1923, ...}

    print("  Stored drift values from Phase 2:")
    for key, val in p2_drifts.items():
        print(f"    {key}: {val:.4f} L2 units")
    print()

    # ── 2. Initialise controller and score memory ─────────────────────────
    print("[2/5] Initialising Dynamic γ Controller...")
    controller   = DynamicGammaController(
        ref_drift=REF_DRIFT, gamma_min=GAMMA_MIN, eta=ETA
    )
    score_memory = ScoreMemory(max_per_class=SCORE_MEMORY_SIZE_PER_CLASS)

    # ── 3. Build DataLoaders ───────────────────────────────────────────────
    print("[3/5] Building CIL DataLoaders (calibration splits only)...")
    task_data = build_cil_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        verify_images=False,
    )

    # ── 4. Sequential processing — one task at a time ─────────────────────
    print("\n[4/5] Processing tasks sequentially...\n")
    quantile_log    = {}    # task_id → {cls_name: q̂_c}
    gamma_log       = {}    # task_id → γₜ for this task's transition
    all_corrections = []

    for task_id in range(4):
        task_classes = TASK_PARTITION[task_id]
        print(f"  ── Task {task_id}: {task_classes} ──")

        # ── 4a. Compute dynamic γ BEFORE processing this task ────────────
        if task_id > 0:
            # Drift from previous task's prototype to this task's prototype
            delta_mu = torch.norm(
                prototypes[task_id] - prototypes[task_id - 1]
            ).item()

            gamma_t = controller.update(delta_mu)
            gamma_log[task_id] = gamma_t

            delta_hat = delta_mu / REF_DRIFT
            print(f"    Δμ(T{task_id-1}→T{task_id}) = {delta_mu:.6f}  "
                  f"Δ̂ = {delta_hat:.4f}  "
                  f"γ_{task_id} = {gamma_t:.4f}")

            # ── 4b. Apply drift correction to all existing scores ─────────
            n_corrected, correction_val = score_memory.apply_drift_correction(
                delta_mu=delta_mu,
                correction_strength=CORRECTION_STRENGTH,
                ref_drift=REF_DRIFT,
            )
            all_corrections.append({
                "task": task_id,
                "delta_mu": round(delta_mu, 6),
                "gamma": round(gamma_t, 4),
                "correction_added": round(correction_val, 6),
                "entries_corrected": n_corrected,
            })
            print(f"    Score correction: +{correction_val:.6f} "
                  f"applied to {n_corrected} entries")
        else:
            print(f"    Task 0: no drift correction (first task)")
            gamma_log[0] = 1.0

        # ── 4c. Load model for this task ──────────────────────────────────
        print(f"    Loading Task {task_id} checkpoint...")
        model = load_task_model(task_id)

        # ── 4d. Run calibration set through model ─────────────────────────
        print(f"    Extracting (scores, latents) from calibration set...")
        cal_loader = task_data[task_id]["calibration"]

        n_added = 0
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(cal_loader):
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                logits, latents = model(images)

                # Compute per-class APS scores for positive samples
                scores_dict = compute_aps_scores_multilabel(
                    logits, labels, task_classes
                )

                # Add to Score Memory
                score_memory.add(
                    scores_dict=scores_dict,
                    latents=latents,
                    labels=labels,
                    task_classes=task_classes,
                    task_id=task_id,
                )
                n_added += sum(len(v) for v in scores_dict.values())

        print(f"    Added {n_added} score-latent pairs to memory")
        print(f"    Score Memory total: {score_memory.total_entries} entries")
        print(f"    Per-class distribution: {score_memory.class_summary()}")

        # ── 4e. Compute Mondrian quantiles q̂_c for this task ─────────────
        print(f"\n    Computing Mondrian quantiles (α={ALPHA}):")
        task_quantiles = {}
        for cls_name in task_classes:
            scores, weights = score_memory.get_scores_and_weights(
                cls_name, task_id, controller
            )
            if len(scores) == 0:
                q_hat = 1.0  # no data → include all
                print(f"      {cls_name:<22}: q̂ = {q_hat:.4f} (NO DATA — degenerate)")
            else:
                q_hat = weighted_quantile(scores, weights, level=1.0 - ALPHA)
                mean_w = float(np.mean(weights))
                min_s  = float(scores.min())
                max_s  = float(scores.max())
                print(f"      {cls_name:<22}: q̂ = {q_hat:.4f}  "
                      f"N={len(scores):3d}  "
                      f"s∈[{min_s:.3f},{max_s:.3f}]  "
                      f"mean_w={mean_w:.4f}")
            task_quantiles[cls_name] = round(q_hat, 6)

        quantile_log[task_id] = task_quantiles

        # Check safety: any quantile = 1.0 indicates degeneracy
        degenerate = [c for c, q in task_quantiles.items() if q >= 0.9999]
        if degenerate:
            print(f"    ⚠ DEGENERATE quantiles (q̂=1.0): {degenerate}")
        else:
            print(f"    [OK] All quantiles are non-degenerate (<1.0)")

        print()

    # ── 5. Controller summary ─────────────────────────────────────────────
    print("[5/5] Dynamic γ Controller Summary:")
    ctrl_summary = controller.summary()
    print(f"  Drift history: {ctrl_summary['drift_history']}")
    print(f"  Gamma history: {ctrl_summary['gamma_history']}")

    # Show accumulated product weights
    print("\n  Accumulated product weights at Task 3 (current):")
    for stored_task in range(4):
        w = controller.get_weight(stored_task, current_task=3)
        n_entries = sum(
            1 for entries in score_memory.memory.values()
            for e in entries if e.task_id == stored_task
        )
        print(f"    Scores from Task {stored_task}: w = {w:.4f}  "
              f"({n_entries} entries in memory)")

    # ── 6. Save everything ────────────────────────────────────────────────
    output = {
        "controller":    ctrl_summary,
        "gamma_log":     {str(k): v for k, v in gamma_log.items()},
        "quantile_log":  {str(k): v for k, v in quantile_log.items()},
        "corrections":   all_corrections,
        "config": {
            "ref_drift":           REF_DRIFT,
            "gamma_min":           GAMMA_MIN,
            "eta":                 ETA,
            "correction_strength": CORRECTION_STRENGTH,
            "alpha":               ALPHA,
            "score_memory_size":   SCORE_MEMORY_SIZE_PER_CLASS,
        },
    }

    json_path = PHASE3_DIR / "phase3_controller.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  [Saved] Controller log → {json_path}")

    # Save score memory as serialisable dict
    mem_save = {}
    for cls_name, entries in score_memory.memory.items():
        mem_save[cls_name] = {
            "scores":   [e.score   for e in entries],
            "task_ids": [e.task_id for e in entries],
            # Save latents stacked as tensor
            "latents":  torch.stack([e.latent for e in entries]),
        }

    mem_path = PHASE3_DIR / "score_memory.pt"
    torch.save({
        "memory":     mem_save,
        "controller": ctrl_summary,
        "quantiles":  quantile_log,
    }, mem_path)
    print(f"  [Saved] Score Memory → {mem_path}")

    print("\n[Phase 3 COMPLETE] — Score Memory and γ Controller ready for Phase 4.\n")
    return output, score_memory, controller


# ── Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--memory_size",  type=int, default=200)
    args = parser.parse_args()

    output, score_memory, controller = run_phase3(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Final printout
    print("\n[Phase 3 Results Summary]")
    print(f"  Controller γ history: {controller.gamma_history}")
    print(f"  Score Memory entries: {score_memory.total_entries}")
    print(f"\n  Final Mondrian Quantiles (Task 3 = last task):")
    for cls, q in output["quantile_log"]["3"].items():
        status = "⚠ DEGENERATE" if q >= 0.9999 else "OK"
        print(f"    {cls:<22}: q̂ = {q:.4f}  [{status}]")