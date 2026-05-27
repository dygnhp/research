"""
Comprehensive experiment runner for the CHM (kanzen) system.

For each of {OX_8, ABC_16, abcd_32}:
    1. Train with autonomous growth
    2. Save summary figure, params, history, growth log
    3. Run robustness sweeps (noise, shift, gamma, ablation)
    4. Compute dataset-variant accuracy on 50 held-out variants per class

Results land in experiments_out/ and are aggregated into a single JSON
summary that the report MD can ingest.
"""
from __future__ import annotations

import os
import json
import time
import numpy as np

# Lightweight matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kanzen import (
    Config, train, save_run,
    classify, accuracy, confusion_matrix,
    noise_sweep, shift_sweep, gamma_sweep, ablation_study,
    load_dataset,
)
from kanzen.preprocess import make_S0
from kanzen.params import assemble_full
from kanzen.viz import summary_figure


# ---------------------------------------------------------------------------
# Per-dataset experiment configuration (tuned for CPU wall time)
# ---------------------------------------------------------------------------
DATASETS = [
    ("OX_8",    2000),     # ~3 min CPU
    ("ABC_16",  5000),     # ~15 min CPU (Phase A: design-spec budget)
    ("abcd_32", 8000),     # ~80 min CPU (Phase A: design-spec budget)
]

OUT_ROOT = "experiments_out_phaseA"


def render_dataset_grid(spec, path):
    """6-per-class grid of canonicals + variants for visual reference."""
    n_classes = len(spec.class_labels)
    n_show = 6
    fig, axes = plt.subplots(n_classes, n_show, figsize=(n_show * 1.5, n_classes * 1.5))
    rng = np.random.RandomState(0)
    for r, lab in enumerate(spec.class_labels):
        canon = spec.canonicals[lab]
        gen = spec.random_generators[lab]
        imgs = [canon] + [gen(rng) for _ in range(n_show - 1)]
        for c, img in enumerate(imgs):
            ax = axes[r, c] if n_classes > 1 else axes[c]
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"'{lab}'", fontsize=14)
            ax.set_title(f"{int(img.sum())}px", fontsize=8)
    fig.suptitle(f"Dataset: {spec.name}  ({n_classes} classes x {n_show} samples)",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [grid] -> {path}")


def run_one(name: str, n_epochs: int) -> dict:
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'=' * 64}\nDATASET {name}  (n_epochs={n_epochs})\n{'=' * 64}")

    cfg = Config.with_dataset(name, n_epochs=n_epochs,
                              log_every=max(50, n_epochs // 30),
                              output_dir=OUT_ROOT)
    print(f"  classes={cfg.n_classes}, K_init={cfg.K_init}, n_max={cfg.n_max}")
    print(f"  peak_lr={cfg.peak_lr}, plateau_window={cfg.plateau_window}")

    # ---- preview the dataset ------------------------------------------------
    render_dataset_grid(cfg.dataset_spec, os.path.join(out_dir, "dataset_preview.png"))

    # ---- training -----------------------------------------------------------
    t0 = time.time()
    run = train(cfg, verbose=True)
    train_time = time.time() - t0
    save_run(out_dir, run)
    state = run["state"]
    print(f"  -- training done in {train_time:.1f}s")

    # ---- per-class canonical trajectories for the summary figure ------------
    w, mu, sigma = assemble_full(state.params, *state.frozen)
    per_class_traj = {}
    canonical_per_class = {}
    spec = cfg.dataset_spec
    for label in cfg.class_labels:
        canon = spec.canonicals[label]
        canonical_per_class[label] = canon
        S0, mask = make_S0(canon, D=state.D, tau=cfg.tau, n_max=cfg.n_max)
        traj = state.simulate_eval(S0, w, mu, sigma)
        per_class_traj[label] = (traj, mask)

    summary_figure(state, cfg, run["history"], run["growth_log"],
                   per_class_traj=per_class_traj,
                   out_path=os.path.join(out_dir, "summary.png"))

    # ---- canonical classification + per-class final distances ---------------
    canonical_results = {}
    for label, img in canonical_per_class.items():
        r = classify(img, state, cfg)
        canonical_results[label] = {
            "pred": r["pred"],
            "distances": {k: round(float(v), 3) for k, v in r["distances"].items()},
            "p_norm": round(float(r["p_norm_mean"]), 3),
        }

    # ---- variant accuracy (full 50-per-class dataset) ----------------------
    dataset_eval = load_dataset(name, n_per_class=50, seed=cfg.dataset_seed + 7)
    var_images, var_labels = [], []
    for lab, imgs in dataset_eval["images_by_label"].items():
        var_images.extend(imgs)
        var_labels.extend([lab] * len(imgs))
    cm = confusion_matrix(var_images, var_labels, state, cfg)
    print(f"  variant accuracy: {cm['overall_acc']:.3f}")

    # ---- noise / shift sweeps per class -------------------------------------
    noise_results = {}
    shift_results = {}
    for label, img in canonical_per_class.items():
        noise_results[label] = noise_sweep(img, label, state, cfg, trials=3)
        ss = shift_sweep(img, label, state, cfg, max_shift=2)
        shift_results[label] = {"acc": ss["acc"], "dxs": ss["dxs"], "dys": ss["dys"],
                                "grid": ss["grid"].tolist()}

    # ---- gamma sweep + ablation --------------------------------------------
    gamma_results = gamma_sweep(canonical_per_class, state, cfg)
    ablation_results = ablation_study(canonical_per_class, state, cfg)

    # ---- aggregate ----------------------------------------------------------
    summary = {
        "dataset": name,
        "n_epochs": n_epochs,
        "train_time_s": round(train_time, 1),
        "final_D": state.D,
        "final_K_learn": state.K_learn,
        "final_loss": round(float(run["history"]["loss"][-1]), 3),
        "n_growth_events": len(run["growth_log"]),
        "growth_events": run["growth_log"],
        "canonical_results": canonical_results,
        "variant_accuracy": {
            "overall":   round(cm["overall_acc"], 3),
            "per_class": {lab: round(cm["per_class_acc"][i], 3)
                          for i, lab in enumerate(cm["labels"])},
            "matrix":    cm["matrix"].tolist(),
            "labels":    list(cm["labels"]),
        },
        "noise_sweep":  {lab: {"levels": list(map(int, r["levels"])),
                               "acc": list(map(float, r["acc"]))}
                         for lab, r in noise_results.items()},
        "shift_sweep":  {lab: {"acc": r["acc"], "grid": r["grid"]}
                         for lab, r in shift_results.items()},
        "gamma_sweep":  {"gammas": gamma_results["gammas"],
                         "acc":    list(map(float, gamma_results["acc"]))},
        "ablation":     {k: {"acc": v["acc"]} for k, v in ablation_results.items()},
        "final_diagnostics": (run["history"]["diag"][-1]
                              if run["history"]["diag"] else {}),
    }
    with open(os.path.join(out_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -- summary saved -> {out_dir}/experiment_summary.json")
    return summary


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    all_summaries = {}
    for name, n_epochs in DATASETS:
        all_summaries[name] = run_one(name, n_epochs)
    with open(os.path.join(OUT_ROOT, "all_summaries.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n{'=' * 64}\nALL EXPERIMENTS COMPLETE\n{'=' * 64}")


if __name__ == "__main__":
    main()
