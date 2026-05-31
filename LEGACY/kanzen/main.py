"""
Command-line entry point.

Usage:
    python -m kanzen.main train    [--dataset NAME] [--epochs N]
    python -m kanzen.main evaluate [--dataset NAME]
    python -m kanzen.main demo     [--dataset NAME]

Dataset names: OX_8 (default), ABC_16, abcd_32.

The Config object derives all training defaults from the dataset, so the
typical command for an experiment is simply

    python -m kanzen.main train --dataset ABC_16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np

from .config import Config
from .data import DATASETS
from .train import train, save_run
from .evaluate import (classify, noise_sweep, shift_sweep,
                       gamma_sweep, ablation_study, confusion_matrix)
from .viz import summary_figure


def _run_id(cfg: Config) -> str:
    return time.strftime(f"run_{cfg.dataset}_%Y%m%d_%H%M%S")


def _build_cli_cfg(args) -> Config:
    overrides = {}
    if args.epochs is not None:
        overrides["n_epochs"] = args.epochs
    if args.seed is not None:
        overrides["dataset_seed"] = args.seed
    return Config.with_dataset(args.dataset, **overrides)


def cmd_train(cfg: Config) -> None:
    out_dir = os.path.join(cfg.output_dir, _run_id(cfg))
    print(f"[main] training {cfg.dataset} -> {out_dir}")
    run = train(cfg, verbose=True)
    save_run(out_dir, run)
    state = run["state"]

    # Build per-class canonical trajectories for the summary figure
    from .preprocess import make_S0
    from .params import assemble_full
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

    for lab in cfg.class_labels:
        r = classify(canonical_per_class[lab], state, cfg)
        print(f"[main] canonical {lab} -> {r['pred']}  "
              f"distances={ {k: round(v, 2) for k, v in r['distances'].items()} }")


def cmd_evaluate(cfg: Config) -> None:
    runs = sorted([d for d in os.listdir(cfg.output_dir)
                   if d.startswith(f"run_{cfg.dataset}_")])
    if not runs:
        print(f"[main] no runs for {cfg.dataset}; run 'train' first.")
        sys.exit(1)
    latest = os.path.join(cfg.output_dir, runs[-1])
    print(f"[main] evaluating {latest}")

    import jax.numpy as jnp
    from .train import _build_state
    npz = np.load(os.path.join(latest, "params.npz"))
    D = int(npz["D"])
    state = _build_state(cfg, D)
    state.params = {
        "w": jnp.asarray(npz["w"]),
        "mu": jnp.asarray(npz["mu"]),
        "sigma_raw": jnp.asarray(npz["sigma_raw"]),
    }
    state.K_learn = int(npz["K_learn"])

    spec = cfg.dataset_spec
    canon_per_class = {lab: spec.canonicals[lab] for lab in cfg.class_labels}

    for lab, img in canon_per_class.items():
        r = classify(img, state, cfg)
        print(f"  canonical {lab} -> {r['pred']}")

    print("  gamma sweep ...", gamma_sweep(canon_per_class, state, cfg))
    print("  ablation   ...", ablation_study(canon_per_class, state, cfg))
    # one noise sweep per class
    for lab, img in canon_per_class.items():
        print(f"  noise {lab}  ...", noise_sweep(img, lab, state, cfg)["acc"])


def cmd_demo(cfg: Config) -> None:
    cfg = Config.with_dataset(cfg.dataset, n_epochs=100, log_every=20,
                              min_epochs_before_grow=50, dataset_seed=cfg.dataset_seed)
    cmd_train(cfg)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="kanzen")
    parser.add_argument("command", choices=["train", "evaluate", "demo"])
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        default="OX_8")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = _build_cli_cfg(args)
    {"train": cmd_train, "evaluate": cmd_evaluate, "demo": cmd_demo}[args.command](cfg)


if __name__ == "__main__":
    main()
