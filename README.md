# Med-CPCL: Medical Conformal Prediction for Continual Learning

A research-grade implementation of a novel framework that provides
distribution-free uncertainty guarantees for sequential medical image
classification under representational drift.

**Thesis:** MSc Data Science / AI
**Dataset:** BloodMNIST (MedMNIST v2) + OrganAMNIST
**Hardware:** NVIDIA RTX 3060, Ubuntu Linux

---

## Key Contribution

Med-CPCL integrates:
- **Prototype-Anchored Quantile Estimator**: time-weighted non-conformity scores
- **Backbone Drift Compensation**: adjusts stale calibration scores using L2 drift
- **Mondrian CP**: per-class coverage guarantees
- **APS Scoring**: adaptive prediction sets for efficiency

---

## Results (Mean +/- Std, 3 seeds: 42, 123, 999)

### Accuracy Metrics

| Method            | AA                                       | BWT                                       | FM                                       |
|-------------------|------------------------------------------|-------------------------------------------|------------------------------------------|
| Fine-Tuning       | 0.250+/-0.000       | -0.986+/-0.000      | 0.986+/-0.000      |
| EWC               | 0.250+/-0.000               | -0.985+/-0.002              | 0.985+/-0.002              |
| Experience Replay | 0.611+/-0.021 | -0.501+/-0.028| 0.501+/-0.028|
| **Med-CPCL**      | 0.610+/-0.019          | -0.502+/-0.026         | 0.502+/-0.026         |

### Conformal Prediction Metrics (Med-CPCL, after Task 3)

| Method        | Wt Min Coverage              | Wt Avg Set Size               | Std Avg Coverage              |
|---------------|------------------------------|-------------------------------|-------------------------------|
| Standard CP   | N/A                          | 4.828+/-0.317 | 0.995+/-0.002|
| Med-CPCL (Wt) | 0.929+/-0.012| 6.876+/-0.035  | 0.982+/-0.003 |

---

## Novel Findings

1. **EWC Fisher Degeneracy (std=0.000)**: Fisher Information collapses to
   near-zero under high-confidence sequential training, rendering EWC
   identical to Fine-Tuning across ALL seeds. Systematic failure, not noise.

2. **Standard CP Efficiency Failure**: Calibrating once on Task 0 causes
   prediction sets to degenerate to near-full-set on new tasks.

3. **Drift Compensation is Critical**: Ablation A1 shows removing drift
   correction drops minimum coverage from 0.934 to 0.884 (below 90%).

4. **Coverage-Accuracy Independence**: Med-CPCL matches ER accuracy
   (0.610 vs 0.611) with zero accuracy cost from the conformal layer.

---

## Setup
```bash
conda create -n medcpcl python=3.10 -y
conda activate medcpcl
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install avalanche-lib==0.4.0 pytorchcv==0.0.67 medmnist==3.0.1
pip install crepes==0.6.0 scikit-learn==1.4.2 numpy==1.26.4
pip install matplotlib==3.8.4 seaborn==0.13.2 pandas==2.2.2
pip install tqdm==4.66.4 wandb==0.17.0 tensorboard==2.16.2
```

---

## Reproduce All Results
```bash
cd ~/medcpcl
conda activate medcpcl

# Full pipeline (~3-4 hours)
python main.py

# Fast run (~30 min, skip multi-seed and ablations)
python main.py --skip-multiseed --skip-ablations

# Individual components
python experiments/run_finetuning.py     # Baseline 1
python experiments/run_replay.py         # Baseline 2
python experiments/run_ewc.py            # Baseline 3
python experiments/run_medcpcl.py        # Proposed method
python experiments/run_ablations.py      # Ablation studies
python experiments/run_multiseed.py      # Statistical validation
python experiments/run_organamnist.py    # Second dataset
python evaluation/visualize.py           # Generate figures
```

---

## Project Structure
```
medcpcl/
├── config.py                       # All hyperparameters (single source of truth)
├── main.py                         # Full pipeline runner
├── data/
│   └── data_loader.py              # BloodMNIST + OrganAMNIST CIL splits
├── models/
│   └── model.py                    # ResNet-18, returns (logits, latent_z)
├── conformal/
│   ├── conformal.py                # StandardCP + APS scoring
│   ├── scoring.py                  # ScoreMemory: stores (si, zi, ti)
│   ├── weighted_cp.py              # WeightedCP + Mondrian CP
│   └── calibration_baselines.py   # Temperature Scaling + Platt Scaling
├── training/
│   ├── train.py                    # Base trainer + hooks
│   ├── replay_buffer.py            # Reservoir sampling buffer
│   └── plugin.py                   # MedCPCLPlugin + drift compensation
├── experiments/
│   ├── run_finetuning.py           # Baseline: Naive Fine-Tuning
│   ├── run_replay.py               # Baseline: Experience Replay
│   ├── run_ewc.py                  # Baseline: EWC
│   ├── run_standard_cp.py          # Baseline: Standard CP
│   ├── run_medcpcl.py              # Proposed: Med-CPCL
│   ├── run_ablations.py            # Ablation studies (A1-A3)
│   ├── run_multiseed.py            # Multi-seed statistical evaluation
│   └── run_organamnist.py          # Second dataset generalisability
└── evaluation/
    ├── visualize.py                # Figures 1-5
    └── visualize_ablations.py      # Figure 6 (ablations)
```

---

## Key Hyperparameters

| Parameter          | Value | Description                          |
|--------------------|-------|--------------------------------------|
| NUM_TASKS          | 4     | Sequential tasks (2 classes each)    |
| NUM_EPOCHS         | 10    | Training epochs per task             |
| REPLAY_BUFFER_SIZE | 20    | Samples per class in replay buffer   |
| ALPHA              | 0.1   | CP miscoverage (target = 90%)        |
| GAMMA              | 0.9   | Time-decay: wi = gamma^(t-ti)        |
| SCORE_MEMORY_SIZE  | 50    | Max calibration scores per class     |
| LATENT_DIM         | 512   | ResNet-18 penultimate layer dim      |

---

## Citation
```
@mastersthesis{medcpcl2025,
  title  = {Med-CPCL: Medical Conformal Prediction for Continual Learning},
  author = {[Your Name]},
  school = {[Your Institution]},
  year   = {2025}
}
```
