# data/data_loader.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
import medmnist
from medmnist import BloodMNIST
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, IMAGE_SIZE, NUM_TASKS, CLASSES_PER_TASK,
    NUM_CLASSES, BATCH_SIZE, SEED
)

torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

TASK_CLASS_MAP = {
    task_id: list(range(task_id * CLASSES_PER_TASK, (task_id + 1) * CLASSES_PER_TASK))
    for task_id in range(NUM_TASKS)
}

def _get_class_indices(dataset, class_list):
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    if labels.ndim > 1:
        labels = labels.squeeze()
    mask = np.isin(labels, class_list)
    return np.where(mask)[0]

def get_task_loaders(task_id: int, cal_fraction: float = 0.2):
    assert 0 <= task_id < NUM_TASKS, f"task_id must be 0-{NUM_TASKS-1}"
    class_list = TASK_CLASS_MAP[task_id]

    train_full = BloodMNIST(split="train", transform=TRAIN_TRANSFORM,
                            download=True, root=DATA_DIR)
    test_full  = BloodMNIST(split="test",  transform=TEST_TRANSFORM,
                            download=True, root=DATA_DIR)

    train_idx = _get_class_indices(train_full, class_list)
    test_idx  = _get_class_indices(test_full,  class_list)

    train_subset = Subset(train_full, train_idx)
    test_subset  = Subset(test_full,  test_idx)

    n_cal   = max(1, int(len(train_subset) * cal_fraction))
    n_train = len(train_subset) - n_cal
    train_split, cal_split = random_split(
        train_subset, [n_train, n_cal],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    cal_loader   = DataLoader(cal_split,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, cal_loader, test_loader

def get_all_test_loader():
    test_full = BloodMNIST(split="test", transform=TEST_TRANSFORM,
                           download=True, root=DATA_DIR)
    return DataLoader(test_full, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=2, pin_memory=True)

def inspect_dataset():
    train_full = BloodMNIST(split="train", download=True,
                            root=DATA_DIR, transform=TRAIN_TRANSFORM)
    test_full  = BloodMNIST(split="test",  download=True,
                            root=DATA_DIR, transform=TEST_TRANSFORM)

    info        = medmnist.INFO["bloodmnist"]
    class_names = list(info["label"].values())

    train_labels = np.array([train_full[i][1] for i in range(len(train_full))]).squeeze()
    test_labels  = np.array([test_full[i][1]  for i in range(len(test_full))]).squeeze()

    print("=" * 60)
    print(f"  BloodMNIST  |  {len(train_full)} train  |  {len(test_full)} test")
    print("=" * 60)
    print(f"  {'Class':<20} {'Train':>8} {'Test':>8}  {'Task':>6}")
    print("-" * 60)
    for cls_id, name in enumerate(class_names):
        tr   = int((train_labels == cls_id).sum())
        te   = int((test_labels  == cls_id).sum())
        task = cls_id // CLASSES_PER_TASK
        print(f"  {name:<20} {tr:>8} {te:>8}  T{task:>5}")
    print("=" * 60)

    print("\nTask -> Class mapping:")
    for t, classes in TASK_CLASS_MAP.items():
        names = [class_names[c] for c in classes]
        print(f"  Task {t}: classes {classes}  ->  {names}")

    print("\nPer-task loader sizes (train | cal | test):")
    for t in range(NUM_TASKS):
        tr_l, cal_l, te_l = get_task_loaders(t)
        print(f"  Task {t}: train={len(tr_l.dataset):>5}  cal={len(cal_l.dataset):>4}  test={len(te_l.dataset):>5}")

if __name__ == "__main__":
    inspect_dataset()
