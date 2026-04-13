"""
Phase 1: NIH Chest X-ray 14 — Patient-wise CIL Data Pipeline
Med-CPCL Extended Framework
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────
NIH_ROOT  = Path("/home/bs_thesis/Documents/NIHChest")
IMAGE_DIR = NIH_ROOT / "images-224" / "images-224"
CSV_PATH  = NIH_ROOT / "Data_Entry_2017.csv"
CACHE_DIR = Path("./cache/phase1")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── CIL Task Partition — 4 tasks, 14 labels ───────────────────────────────
TASK_PARTITION = {
    0: ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration"],
    1: ["Pneumonia", "Pneumothorax", "Consolidation", "Edema"],
    2: ["Emphysema", "Fibrosis", "Pleural_Thickening", "Mass"],
    3: ["Nodule", "Hernia"],
}

ALL_LABELS = []
for labels in TASK_PARTITION.values():
    ALL_LABELS.extend(labels)

assert len(ALL_LABELS) == 14, f"Expected 14 labels, got {len(ALL_LABELS)}"
assert len(set(ALL_LABELS)) == 14, "Duplicate labels found in task partition!"

LABEL_TO_IDX = {label: idx for idx, label in enumerate(ALL_LABELS)}
TASK_LABEL_IDX = {
    task_id: [LABEL_TO_IDX[l] for l in labels]
    for task_id, labels in TASK_PARTITION.items()
}

# ── ImageNet Normalisation (for pretrained ResNet-50) ─────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(split: str) -> T.Compose:
    """
    Images are pre-resized to 224×224, so no Resize step is needed.
    Train split uses a light crop + flip + colour jitter for augmentation.
    Val / test / calibration use only ToTensor + Normalize.
    """
    if split == "train":
        return T.Compose([
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val / test / calibration
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ── Dataset Class ─────────────────────────────────────────────────────────
class NIHChestXrayDataset(Dataset):
    """
    Multi-label NIH CXR-14 Dataset for a SINGLE CIL task.

    Returns:
        image  : Tensor [3, 224, 224]
        label  : Tensor [num_task_classes] — binary multi-hot for task classes only
        patient: str — patient ID (for verification)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        task_id: int,
        split: str,
        image_dir: Path,
    ):
        self.df          = dataframe.reset_index(drop=True)
        self.task_id     = task_id
        self.task_labels = TASK_PARTITION[task_id]
        self.transform   = get_transforms(split)
        self.image_dir   = image_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Load image
        img_path = self.image_dir / row["Image Index"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Build multi-hot label vector for THIS task's classes only
        finding_labels = str(row["Finding Labels"]).split("|")
        label = torch.zeros(len(self.task_labels), dtype=torch.float32)
        for i, cls_name in enumerate(self.task_labels):
            if cls_name in finding_labels:
                label[i] = 1.0

        return image, label, str(row["Patient ID"])


# ── Data Splitting ─────────────────────────────────────────────────────────
def build_patient_wise_splits(
    csv_path: Path,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    # test_ratio = 1 - train_ratio - val_ratio = 0.10
) -> tuple[list, list, list]:
    """
    Performs patient-wise 80/10/10 split to prevent data leakage.
    All images from the same patient stay in the same split.

    Returns:
        train_patients, val_patients, test_patients : lists of patient IDs
    """
    cache_file = CACHE_DIR / "patient_splits.json"

    if cache_file.exists():
        print(f"[Phase 1] Loading cached patient splits from {cache_file}")
        with open(cache_file) as f:
            splits = json.load(f)
        return splits["train"], splits["val"], splits["test"]

    print("[Phase 1] Building patient-wise splits...")
    df = pd.read_csv(csv_path)

    all_patients = df["Patient ID"].unique().tolist()
    print(f"  Total unique patients: {len(all_patients):,}")
    print(f"  Total images:          {len(df):,}")

    # First split: train vs (val + test)
    train_patients, temp_patients = train_test_split(
        all_patients,
        test_size=(1.0 - train_ratio),
        random_state=SEED,
    )

    # Second split: val vs test from the remainder
    val_ratio_adjusted = val_ratio / (1.0 - train_ratio)
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=(1.0 - val_ratio_adjusted),
        random_state=SEED,
    )

    print(f"  Train patients: {len(train_patients):,}")
    print(f"  Val patients:   {len(val_patients):,}")
    print(f"  Test patients:  {len(test_patients):,}")

    # Verify no overlap
    assert len(set(train_patients) & set(val_patients))  == 0, "LEAKAGE: train/val overlap!"
    assert len(set(train_patients) & set(test_patients)) == 0, "LEAKAGE: train/test overlap!"
    assert len(set(val_patients)   & set(test_patients)) == 0, "LEAKAGE: val/test overlap!"
    print("  [OK] Zero patient overlap across all splits.")

    splits = {"train": train_patients, "val": val_patients, "test": test_patients}
    with open(cache_file, "w") as f:
        json.dump(splits, f)

    return train_patients, val_patients, test_patients


def filter_df_for_task(df: pd.DataFrame, task_id: int) -> pd.DataFrame:
    """
    Returns rows where at least one label belongs to the given task,
    OR the image is labelled 'No Finding' (negative examples).
    """
    task_classes = set(TASK_PARTITION[task_id])

    def row_relevant(finding_str):
        findings = set(str(finding_str).split("|"))
        return bool(findings & task_classes) or finding_str == "No Finding"

    mask = df["Finding Labels"].apply(row_relevant)
    return df[mask].copy()


def build_calibration_split(
    train_df: pd.DataFrame,
    cal_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits task training data into 80% train / 20% calibration
    using patient-wise splitting to maintain integrity.
    """
    patient_ids = train_df["Patient ID"].unique().tolist()
    pure_train_patients, cal_patients = train_test_split(
        patient_ids, test_size=cal_ratio, random_state=SEED
    )
    pure_train_df = train_df[train_df["Patient ID"].isin(pure_train_patients)]
    cal_df        = train_df[train_df["Patient ID"].isin(cal_patients)]
    return pure_train_df.reset_index(drop=True), cal_df.reset_index(drop=True)


# ── Main Pipeline ─────────────────────────────────────────────────────────
def build_cil_dataloaders(
    batch_size:    int  = 32,
    num_workers:   int  = 4,
    verify_images: bool = False,
) -> dict:
    """
    Builds complete CIL dataloader structure for all 4 tasks.

    Returns:
        task_data[task_id] = {
            "train":       DataLoader,
            "calibration": DataLoader,
            "val":         DataLoader,
            "test":        DataLoader,
            "stats":       dict  (class counts, pos_weights)
        }
    """
    print("\n" + "="*60)
    print("  Phase 1: NIH CXR-14 — CIL Data Pipeline")
    print("="*60)

    # 1. Load master CSV
    print("\n[1/5] Loading Data_Entry_2017.csv...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df):,} rows, {df['Patient ID'].nunique():,} unique patients.")

    # 2. Patient-wise split
    print("\n[2/5] Building patient-wise splits...")
    train_patients, val_patients, test_patients = build_patient_wise_splits(CSV_PATH)

    train_df = df[df["Patient ID"].isin(train_patients)].copy()
    val_df   = df[df["Patient ID"].isin(val_patients)].copy()
    test_df  = df[df["Patient ID"].isin(test_patients)].copy()

    print(f"  Train images: {len(train_df):,}")
    print(f"  Val images:   {len(val_df):,}")
    print(f"  Test images:  {len(test_df):,}")

    # 3. Optionally verify a sample of images exist on disk
    if verify_images:
        print("\n[3/5] Verifying image files...")
        sample = df.sample(min(500, len(df)), random_state=SEED)
        missing = [
            r["Image Index"]
            for _, r in sample.iterrows()
            if not (IMAGE_DIR / r["Image Index"]).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image(s) not found. First missing: {missing[0]}\n"
                f"Check IMAGE_DIR = {IMAGE_DIR}"
            )
        print(f"  [OK] Verified {len(sample)} sample images — all present.")
    else:
        print("\n[3/5] Skipping image verification (set verify_images=True to enable).")

    # 4. Build per-task datasets + dataloaders
    print("\n[4/5] Building per-task datasets...")
    task_data = {}

    for task_id in range(4):
        print(f"\n  ── Task {task_id}: {TASK_PARTITION[task_id]} ──")

        # Filter to task-relevant rows
        task_train_df = filter_df_for_task(train_df, task_id)
        task_val_df   = filter_df_for_task(val_df,   task_id)
        task_test_df  = filter_df_for_task(test_df,  task_id)

        # Calibration split from training data (patient-wise)
        pure_train_df, cal_df = build_calibration_split(task_train_df, cal_ratio=0.20)

        print(f"    Pure train:   {len(pure_train_df):,} images")
        print(f"    Calibration:  {len(cal_df):,} images")
        print(f"    Val:          {len(task_val_df):,} images")
        print(f"    Test:         {len(task_test_df):,} images")

        # Build datasets
        train_ds = NIHChestXrayDataset(pure_train_df, task_id, "train", IMAGE_DIR)
        cal_ds   = NIHChestXrayDataset(cal_df,        task_id, "val",   IMAGE_DIR)
        val_ds   = NIHChestXrayDataset(task_val_df,   task_id, "val",   IMAGE_DIR)
        test_ds  = NIHChestXrayDataset(task_test_df,  task_id, "test",  IMAGE_DIR)

        # Compute class frequencies for BCE pos_weight
        n_classes  = len(TASK_PARTITION[task_id])
        pos_counts = np.zeros(n_classes)
        total      = len(pure_train_df)

        for i, label_name in enumerate(TASK_PARTITION[task_id]):
            pos_counts[i] = pure_train_df["Finding Labels"].apply(
                lambda x, ln=label_name: ln in str(x).split("|")
            ).sum()

        neg_counts  = total - pos_counts
        pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32)
        print(f"    pos_weights:  {pos_weights.numpy().round(2).tolist()}")

        # Build DataLoaders
        task_data[task_id] = {
            "train": DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=True,
            ),
            "calibration": DataLoader(
                cal_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            ),
            "val": DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            ),
            "test": DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            ),
            "stats": {
                "train_size":  len(pure_train_df),
                "cal_size":    len(cal_df),
                "val_size":    len(task_val_df),
                "test_size":   len(task_test_df),
                "pos_weights": pos_weights,
                "n_classes":   n_classes,
                "class_names": TASK_PARTITION[task_id],
            },
        }

    # 5. Final summary
    print("\n[5/5] Summary")
    print("="*60)
    for task_id, data in task_data.items():
        s = data["stats"]
        print(f"  Task {task_id} {s['class_names']}")
        print(f"    train={s['train_size']:,}  cal={s['cal_size']:,}  "
              f"val={s['val_size']:,}  test={s['test_size']:,}")
    print("="*60)
    print("[Phase 1 COMPLETE] — All DataLoaders built successfully.\n")

    return task_data


# ── Verification Helpers ──────────────────────────────────────────────────
def verify_one_batch(task_data: dict, task_id: int = 0) -> None:
    """
    Pulls one batch from each loader and prints tensor shapes.
    Use this to confirm everything is wired correctly.
    """
    print(f"\n[Verification] Checking batch shapes for Task {task_id}...")
    for split in ["train", "calibration", "val", "test"]:
        loader = task_data[task_id][split]
        images, labels, patients = next(iter(loader))
        print(f"  {split:14s} — images: {list(images.shape)} "
              f"labels: {list(labels.shape)} "
              f"dtype_img: {images.dtype} "
              f"dtype_lbl: {labels.dtype}")
        assert images.min() < 0, "Images should be normalised (some values < 0)"
        assert labels.max() <= 1.0 and labels.min() >= 0.0, \
            "Labels should be binary [0, 1]"
    print("  [OK] All batch shapes and dtypes correct.\n")


def print_label_distribution(task_data: dict) -> None:
    """Prints positive label prevalence for all tasks (quality check)."""
    print("\n[Verification] Label Prevalence in Training Sets:")
    print(f"  {'Task':<6} {'Label':<22} {'Positives':>10} {'Total':>10} {'Prevalence':>12}")
    print("  " + "-"*64)
    for task_id, data in task_data.items():
        stats = data["stats"]
        for i, cls_name in enumerate(stats["class_names"]):
            pw    = stats["pos_weights"][i].item()
            total = stats["train_size"]
            pos   = int(total / (pw + 1))
            prevalence = pos / total * 100
            marker = " ← RARE" if prevalence < 1.0 else ""
            print(f"  T{task_id:<5} {cls_name:<22} {pos:>10,} {total:>10,} "
                  f"{prevalence:>10.2f}%{marker}")
    print()


# ── Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",    type=int,  default=32)
    parser.add_argument("--num_workers",   type=int,  default=4)
    parser.add_argument("--verify_images", action="store_true",
                        help="Check image files exist on disk (slower)")
    args = parser.parse_args()

    task_data = build_cil_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        verify_images=args.verify_images,
    )

    verify_one_batch(task_data, task_id=0)
    verify_one_batch(task_data, task_id=3)   # Task 3 has Hernia (rare label)

    print_label_distribution(task_data)

    print("[Phase 1 SUCCESS] Ready for Phase 2: Backbone Training.\n")