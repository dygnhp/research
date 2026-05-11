"""
Command-line entry point.

Usage:
    python -m kanzen.main train       # full training run with autonomous growth
    python -m kanzen.main evaluate    # evaluate the most recent run
    python -m kanzen.main demo        # short demo run (few epochs)

All options are read from the Config class.  Override via environment
variables or by editing the Config defaults.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

from .config import Config
from .data import O_CANONICAL, X_CANONICAL
from .train import train, save_run
from .evaluate import (classify, noise_sweep, shift_sweep,
                       gamma_sweep, ablation_study)
from .viz import summary_figure


def _run_id() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


def cmd_train(cfg: Config) -> None:
    out_dir = os.path.join(cfg.output_dir, _run_id())
    print(f"[main] training run -> {out_dir}")
    run = train(cfg, verbose=True)
    save_run(out_dir, run)

    state = run["state"]
    from .preprocess import make_S0
    from .params import assemble_full
    S0_O, mask_O = make_S0(O_CANONICAL, D=state.D, tau=cfg.tau, n_max=cfg.n_max)
    S0_X, mask_X = make_S0(X_CANONICAL, D=state.D, tau=cfg.tau, n_max=cfg.n_max)
    w, mu, sigma = assemble_full(state.params, *state.frozen)
    traj_O = state.simulate_eval(S0_O, w, mu, sigma)
    traj_X = state.simulate_eval(S0_X, w, mu, sigma)

    summary_figure(
        state, cfg, run["history"], run["growth_log"],
        trajectory_O=traj_O, trajectory_X=traj_X,
        mask_O=mask_O, mask_X=mask_X,
        out_path=os.path.join(out_dir, "summary.png"),
    )

    rO = classify(O_CANONICAL, state, cfg)
    rX = classify(X_CANONICAL, state, cfg)
    print(f"[main] canonical O -> {rO['pred']}  (d_O={rO['d_O']:.2f}, d_X={rO['d_X']:.2f})")
    print(f"[main] canonical X -> {rX['pred']}  (d_O={rX['d_O']:.2f}, d_X={rX['d_X']:.2f})")


def cmd_evaluate(cfg: Config) -> None:
    """Evaluate the most recent training run."""
    runs = sorted([d for d in os.listdir(cfg.output_dir)
                   if d.startswith("run_")])
    if not runs:
        print("[main] no runs found; run 'train' first.")
        sys.exit(1)
    latest = os.path.join(cfg.output_dir, runs[-1])
    print(f"[main] evaluating {latest}")

    # Reload params
    import jax.numpy as jnp
    npz = np.load(os.path.join(latest, "params.npz"))
    D = int(npz["D"])
    from .train import _build_state
    state = _build_state(cfg, D)
    state.params = {
        "w": jnp.asarray(npz["w"]),
        "mu": jnp.asarray(npz["mu"]),
        "sigma_raw": jnp.asarray(npz["sigma_raw"]),
    }
    state.K_learn = int(npz["K_learn"])

    rO = classify(O_CANONICAL, state, cfg)
    rX = classify(X_CANONICAL, state, cfg)
    print(f"  canonical O -> {rO['pred']}")
    print(f"  canonical X -> {rX['pred']}")

    print("  noise sweep (X)  ...", noise_sweep(X_CANONICAL, "X", state, cfg))
    print("  shift sweep (O)  ...", shift_sweep(O_CANONICAL, "O", state, cfg)["acc"])
    print("  gamma sweep      ...", gamma_sweep(O_CANONICAL, X_CANONICAL, state, cfg))
    print("  ablation         ...", ablation_study(O_CANONICAL, X_CANONICAL, state, cfg))


def cmd_demo(cfg: Config) -> None:
    """Tiny run for end-to-end smoke test (~100 epochs)."""
    cfg.n_epochs = 100
    cfg.log_every = 20
    cfg.min_epochs_before_grow = 50
    cmd_train(cfg)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "train"
    cfg = Config()
    if cmd == "train":
        cmd_train(cfg)
    elif cmd == "evaluate":
        cmd_evaluate(cfg)
    elif cmd == "demo":
        cmd_demo(cfg)
    else:
        print(f"unknown command: {cmd}")
        print("usage: python -m kanzen.main [train|evaluate|demo]")
        sys.exit(1)


if __name__ == "__main__":
    main()
