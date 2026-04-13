# main.py
# Runs the complete Med-CPCL experimental pipeline in sequence.
# Usage:
#   python main.py                  # full pipeline
#   python main.py --skip-multiseed # skip slow multi-seed run
#   python main.py --only baseline  # run only baselines

import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def banner(title):
    print("\n" + "#" * 65)
    print(f"#  {title}")
    print("#" * 65)

def run_step(label, fn, skip=False):
    if skip:
        print(f"  [SKIPPED] {label}")
        return None
    banner(label)
    t0 = time.time()
    result = fn()
    print(f"  Completed in {time.time()-t0:.1f}s")
    return result

def main():
    parser = argparse.ArgumentParser(description="Med-CPCL full pipeline")
    parser.add_argument("--skip-multiseed", action="store_true",
                        help="Skip multi-seed evaluation (saves ~2hrs)")
    parser.add_argument("--skip-ablations", action="store_true",
                        help="Skip ablation studies (saves ~35min)")
    parser.add_argument("--only", type=str, default=None,
                        choices=["baselines","cp","medcpcl","ablations",
                                 "multiseed","figures"],
                        help="Run only one section")
    args = parser.parse_args()

    only = args.only
    t_start = time.time()

    print("=" * 65)
    print("  Med-CPCL: Full Experimental Pipeline")
    print("  Medical Conformal Prediction for Continual Learning")
    print("=" * 65)

    # ── Baselines ──────────────────────────────────────────────────────────
    if only in (None, "baselines"):
        from experiments.run_finetuning import run_finetuning
        from experiments.run_replay     import run_replay
        from experiments.run_ewc        import run_ewc

        run_step("Baseline 1: Fine-Tuning",       run_finetuning)
        run_step("Baseline 2: Experience Replay",  run_replay)
        run_step("Baseline 3: EWC",                run_ewc)

    # ── Calibration + Standard CP ──────────────────────────────────────────
    if only in (None, "cp"):
        import torch
        from config import CKPT_DIR, DEVICE
        from models.model import build_model
        from conformal.calibration_baselines import run_calibration_baselines
        from experiments.run_standard_cp import run_standard_cp

        def cal_step():
            model = build_model()
            ckpt  = os.path.join(CKPT_DIR, "replay_final.pt")
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            return run_calibration_baselines(model)

        run_step("Phase 7: Calibration Baselines",     cal_step)
        run_step("Phase 8: Standard CP (baseline)",    run_standard_cp)

    # ── Med-CPCL ───────────────────────────────────────────────────────────
    if only in (None, "medcpcl"):
        from experiments.run_medcpcl import run_medcpcl
        run_step("Phase 9: Med-CPCL (proposed method)", run_medcpcl)

    # ── Ablations ──────────────────────────────────────────────────────────
    if only in (None, "ablations"):
        from experiments.run_ablations import run_ablations
        run_step("Phase 11: Ablation Studies",
                 run_ablations,
                 skip=args.skip_ablations)

    # ── Multi-Seed ─────────────────────────────────────────────────────────
    if only in (None, "multiseed"):
        from experiments.run_multiseed import run_multiseed
        run_step("Phase 12: Multi-Seed Evaluation",
                 run_multiseed,
                 skip=args.skip_multiseed)

    # ── Figures ────────────────────────────────────────────────────────────
    if only in (None, "figures"):
        from evaluation.visualize          import (fig1_accuracy_matrices,
                                                    fig2_metric_comparison,
                                                    fig3_cp_comparison,
                                                    fig4_ece_comparison,
                                                    fig5_fisher_degeneracy)
        from evaluation.visualize_ablations import main as abl_fig

        def gen_figures():
            print("  Generating all thesis figures...")
            fig1_accuracy_matrices()
            fig2_metric_comparison()
            fig3_cp_comparison()
            fig4_ece_comparison()
            fig5_fisher_degeneracy()
            abl_fig() if hasattr(abl_fig, "__call__") else None

        run_step("Phase 10+: Generate All Figures", gen_figures)

    total = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"  Pipeline complete in {total/60:.1f} minutes")
    print(f"  Results: ~/medcpcl/results/tables/")
    print(f"  Figures: ~/medcpcl/results/figures/")
    print("=" * 65)


if __name__ == "__main__":
    main()
