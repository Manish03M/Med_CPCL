# training/replay_buffer.py
# Reservoir-sampling replay buffer.
#
# Thesis mapping:
#   - "Replay buffer (training)" in Memory Separation requirement
#   - Stores raw (x, y) samples -- separate from Score Memory (si, zi, ti)
#   - Uses reservoir sampling for unbiased class representation

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import REPLAY_BUFFER_SIZE, BATCH_SIZE, SEED

rng = np.random.default_rng(SEED)


class ReplayBuffer:
    """
    Fixed-size replay buffer using reservoir sampling per class.

    Stores up to REPLAY_BUFFER_SIZE samples per class seen so far.
    At training time, returns a DataLoader mixing replay + current task data.
    """

    def __init__(self, max_per_class: int = REPLAY_BUFFER_SIZE):
        self.max_per_class = max_per_class
        # {class_id: {"x": Tensor, "y": Tensor, "count": int}}
        self.buffer = {}

    def update(self, loader: DataLoader, task_id: int):
        """
        Add samples from loader into the buffer using reservoir sampling.
        Called after training each task.
        """
        xs, ys = [], []
        for x, y in loader:
            xs.append(x)
            ys.append(y.squeeze().long())
        xs = torch.cat(xs, dim=0)   # (N, 3, 28, 28)
        ys = torch.cat(ys, dim=0)   # (N,)

        classes = ys.unique().tolist()
        for cls in classes:
            mask   = (ys == cls)
            x_cls  = xs[mask]
            y_cls  = ys[mask]
            n_new  = x_cls.size(0)

            if cls not in self.buffer:
                # First time seeing this class
                idx = rng.choice(n_new,
                                 size=min(self.max_per_class, n_new),
                                 replace=False)
                self.buffer[cls] = {
                    "x"    : x_cls[idx].cpu(),
                    "y"    : y_cls[idx].cpu(),
                    "count": n_new
                }
            else:
                # Reservoir sampling: merge existing + new
                existing = self.buffer[cls]
                all_x = torch.cat([existing["x"], x_cls.cpu()], dim=0)
                all_y = torch.cat([existing["y"], y_cls.cpu()], dim=0)
                total = existing["count"] + n_new
                idx   = rng.choice(len(all_x),
                                   size=min(self.max_per_class, len(all_x)),
                                   replace=False)
                self.buffer[cls] = {
                    "x"    : all_x[idx],
                    "y"    : all_y[idx],
                    "count": total
                }

    def get_loader(self, batch_size: int = BATCH_SIZE):
        """
        Returns a DataLoader of all buffered samples.
        Returns None if buffer is empty.
        """
        if not self.buffer:
            return None
        all_x = torch.cat([v["x"] for v in self.buffer.values()], dim=0)
        all_y = torch.cat([v["y"] for v in self.buffer.values()], dim=0)
        dataset = TensorDataset(all_x, all_y)
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, num_workers=0)

    def get_combined_loader(self, current_loader: DataLoader,
                            batch_size: int = BATCH_SIZE):
        """
        Combines current task data with replay buffer data.
        This is the loader passed to _run_epoch() in ERTrainer.
        """
        # Collect current task data
        xs, ys = [], []
        for x, y in current_loader:
            xs.append(x)
            ys.append(y.squeeze().long())
        curr_x = torch.cat(xs, dim=0)
        curr_y = torch.cat(ys, dim=0)

        # Add replay data if available
        if self.buffer:
            rep_x = torch.cat([v["x"] for v in self.buffer.values()], dim=0)
            rep_y = torch.cat([v["y"] for v in self.buffer.values()], dim=0)
            all_x = torch.cat([curr_x, rep_x.to(curr_x.device)], dim=0)
            all_y = torch.cat([curr_y, rep_y.to(curr_y.device)], dim=0)
        else:
            all_x, all_y = curr_x, curr_y

        dataset = TensorDataset(all_x, all_y)
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, num_workers=0)

    def summary(self):
        """Print buffer contents."""
        total = sum(v["x"].size(0) for v in self.buffer.values())
        print(f"  ReplayBuffer: {len(self.buffer)} classes | {total} total samples")
        for cls, v in self.buffer.items():
            print(f"    Class {cls}: {v['x'].size(0)} samples (seen {v['count']})")
