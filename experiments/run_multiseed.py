# experiments/run_multiseed.py
# Runs all methods across 3 seeds and reports mean +/- std.
# Required for publication-level statistical validity.

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from datetime import datetime

from config import (NUM_TASKS, NUM_EPOCHS, DEVICE, TABLES_DIR,
                    CKPT_DIR, LR, WEIGHT_DECAY, ALPHA)
from models.model import build_model
from training.train import Trainer, evaluate_all_tasks, train_one_epoch
from training.replay_buffer import ReplayBuffer
from training.plugin import MedCPCLPlugin
from conformal.scoring import ScoreMemory
from conformal.weighted_cp import WeightedCP
from conformal.conformal import StandardCP
from data.data_loader import get_task_loaders
from experiments.run_finetuning import compute_metrics
from experiments.run_replay import ERTrainer
from experiments.run_ewc import EWCTrainer

SEEDS = [42, 123, 999]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_finetuning_seed(seed):
    set_seed(seed)
    model   = build_model()
    trainer = Trainer(model)
    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    test_loaders = []
    for task_id in range(NUM_TASKS):
        tr, _, te = get_task_loaders(task_id)
        test_loaders.append(te)
        trainer.train_task(task_id, tr, num_epochs=NUM_EPOCHS, verbose=False)
        res = evaluate_all_tasks(model, test_loaders)
        for j, acc in res.items():
            acc_matrix[task_id][j] = acc
    AA, BWT, FM = compute_metrics(acc_matrix)
    return {"AA": AA, "BWT": BWT, "FM": FM}


def run_replay_seed(seed):
    set_seed(seed)
    model         = build_model()
    replay_buffer = ReplayBuffer()
    trainer       = ERTrainer(model, replay_buffer)
    acc_matrix    = np.zeros((NUM_TASKS, NUM_TASKS))
    test_loaders  = []
    for task_id in range(NUM_TASKS):
        tr, _, te = get_task_loaders(task_id)
        test_loaders.append(te)
        trainer.train_task(task_id, tr, num_epochs=NUM_EPOCHS, verbose=False)
        res = evaluate_all_tasks(model, test_loaders)
        for j, acc in res.items():
            acc_matrix[task_id][j] = acc
    AA, BWT, FM = compute_metrics(acc_matrix)
    return {"AA": AA, "BWT": BWT, "FM": FM}


def run_ewc_seed(seed):
    set_seed(seed)
    model   = build_model()
    trainer = EWCTrainer(model)
    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    test_loaders = []
    for task_id in range(NUM_TASKS):
        tr, _, te = get_task_loaders(task_id)
        test_loaders.append(te)
        trainer.train_task(task_id, tr, num_epochs=NUM_EPOCHS, verbose=False)
        res = evaluate_all_tasks(model, test_loaders)
        for j, acc in res.items():
            acc_matrix[task_id][j] = acc
    AA, BWT, FM = compute_metrics(acc_matrix)
    return {"AA": AA, "BWT": BWT, "FM": FM}


def run_medcpcl_seed(seed):
    set_seed(seed)
    model         = build_model()
    replay_buffer = ReplayBuffer()
    score_memory  = ScoreMemory()
    plugin        = MedCPCLPlugin(score_memory)
    trainer       = ERTrainer(model, replay_buffer)
    wcp           = WeightedCP(score_memory, alpha=ALPHA,
                               gamma=0.9, use_mondrian=True)
    std_cp        = StandardCP(alpha=ALPHA, scoring="aps")

    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    test_loaders = []
    cp_final     = {}

    for task_id in range(NUM_TASKS):
        tr, cal, te = get_task_loaders(task_id)
        test_loaders.append(te)
        trainer.train_task(task_id, tr, num_epochs=NUM_EPOCHS, verbose=False)
        plugin.after_training_task(model, cal, task_id)
        wcp.calibrate(current_task=task_id)
        std_cp.calibrate(model, cal)
        res = evaluate_all_tasks(model, test_loaders)
        for j, acc in res.items():
            acc_matrix[task_id][j] = acc

    for j in range(NUM_TASKS):
        _, _, tl = get_task_loaders(j)
        wt_cov,  wt_sz  = wcp.evaluate(model, tl)
        std_cov, std_sz = std_cp.evaluate(model, tl)
        cp_final[j] = {
            "wt_cov": wt_cov, "wt_sz": wt_sz,
            "std_cov": std_cov, "std_sz": std_sz
        }

    AA, BWT, FM = compute_metrics(acc_matrix)
    avg_wt_cov  = np.mean([cp_final[j]["wt_cov"]  for j in range(NUM_TASKS)])
    min_wt_cov  = min(cp_final[j]["wt_cov"]        for j in range(NUM_TASKS))
    avg_wt_sz   = np.mean([cp_final[j]["wt_sz"]   for j in range(NUM_TASKS)])
    avg_std_cov = np.mean([cp_final[j]["std_cov"] for j in range(NUM_TASKS)])
    avg_std_sz  = np.mean([cp_final[j]["std_sz"]  for j in range(NUM_TASKS)])

    return {
        "AA": AA, "BWT": BWT, "FM": FM,
        "wt_avg_cov": avg_wt_cov, "wt_min_cov": min_wt_cov,
        "wt_avg_sz": avg_wt_sz,
        "std_avg_cov": avg_std_cov, "std_avg_sz": avg_std_sz
    }


def summarise(results, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in results]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def run_multiseed():
    print("=" * 70)
    print("  MULTI-SEED EVALUATION  (seeds: " + str(SEEDS) + ")")
    print("=" * 70)

    methods = {
        "Fine-Tuning"      : (run_finetuning_seed, ["AA","BWT","FM"]),
        "Experience Replay": (run_replay_seed,     ["AA","BWT","FM"]),
        "EWC"              : (run_ewc_seed,        ["AA","BWT","FM"]),
        "Med-CPCL"         : (run_medcpcl_seed,
                              ["AA","BWT","FM",
                               "wt_avg_cov","wt_min_cov","wt_avg_sz",
                               "std_avg_cov","std_avg_sz"]),
    }

    all_summaries = {}

    for name, (fn, keys) in methods.items():
        print(f"\n  [{name}]")
        seed_results = []
        for seed in SEEDS:
            print(f"    seed={seed} ...", end=" ", flush=True)
            r = fn(seed)
            seed_results.append(r)
            print(f"AA={r['AA']:.4f}  BWT={r['BWT']:.4f}")
        summary = summarise(seed_results, keys)
        all_summaries[name] = summary

    # Print final table
    print("\n" + "=" * 70)
    print("  FINAL RESULTS: Mean +/- Std over " + str(len(SEEDS)) + " seeds")
    print("=" * 70)
    print(f"  {'Method':<22} {'AA':>14} {'BWT':>14} {'FM':>14}")
    print("-" * 70)
    for name, s in all_summaries.items():
        aa  = f"{s['AA']['mean']:.3f}+/-{s['AA']['std']:.3f}"
        bwt = f"{s['BWT']['mean']:.3f}+/-{s['BWT']['std']:.3f}"
        fm  = f"{s['FM']['mean']:.3f}+/-{s['FM']['std']:.3f}"
        print(f"  {name:<22} {aa:>14} {bwt:>14} {fm:>14}")
    print("=" * 70)

    print(f"\n  {'Method':<12} {'WtAvgCov':>14} {'WtMinCov':>14} {'WtAvgSz':>12} {'StdAvgCov':>14} {'StdAvgSz':>12}")
    print("-" * 70)
    s = all_summaries["Med-CPCL"]
    def fmt(k):
        return f"{s[k]['mean']:.3f}+/-{s[k]['std']:.3f}"
    print(f"  {'Med-CPCL':<12} {fmt('wt_avg_cov'):>14} {fmt('wt_min_cov'):>14} "
          f"{fmt('wt_avg_sz'):>12} {fmt('std_avg_cov'):>14} {fmt('std_avg_sz'):>12}")
    print("=" * 70)

    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, "multiseed_results.json")
    with open(path, "w") as f:
        json.dump({"seeds": SEEDS,
                   "timestamp": datetime.now().isoformat(),
                   "results": all_summaries}, f, indent=2)
    print(f"\n  Saved to: {path}")
    return all_summaries


if __name__ == "__main__":
    run_multiseed()
