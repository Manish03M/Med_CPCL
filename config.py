# config.py
# Central configuration for Med-CPCL project
# ALL experiments read from this file for reproducibility
import os
# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR   = os.path.join(RESULTS_DIR, "tables")
CKPT_DIR     = os.path.join(RESULTS_DIR, "checkpoints")
LOGS_DIR     = os.path.join(BASE_DIR, "logs")
# ── Dataset ────────────────────────────────────────────────────────────────
DATASET      = "bloodmnist"      # options: bloodmnist, organamnist
IMAGE_SIZE   = 28
NUM_CLASSES  = 8                 # BloodMNIST has 8 classes
NUM_TASKS    = 4                 # 8 classes → 2 classes per task
CLASSES_PER_TASK = 2
# ── Model ──────────────────────────────────────────────────────────────────
BACKBONE     = "resnet18"
PRETRAINED   = False             # MedMNIST is 28×28; ImageNet weights are suboptimal
LATENT_DIM   = 512               # ResNet-18 penultimate layer dimension
# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE   = 64
NUM_EPOCHS   = 10                # per task
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE       = "cuda"            # "cpu" if no GPU
# ── Continual Learning ─────────────────────────────────────────────────────
REPLAY_BUFFER_SIZE  = 20         # samples per class in replay buffer
EWC_LAMBDA          = 0.4
# ── Conformal Prediction ───────────────────────────────────────────────────
ALPHA               = 0.1        # target miscoverage (1-alpha = 90% coverage)
GAMMA               = 0.9        # time-decay factor for wi = gamma^(t - ti)
SCORE_MEMORY_SIZE   = 50         # max calibration scores per class stored
# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
# ── Logging ────────────────────────────────────────────────────────────────
USE_WANDB   = False              # set True when you want W&B tracking
PROJECT_NAME = "med-cpcl"