"""
Forward-only evaluation utilities.

Once training is settled, we use the *evaluation* simulator (no
checkpoint) to characterize the classifier.  The protocol mirrors the
spec's "Step 3" classification (Section 5) plus the standard robustness
suite from Block III (noise, shift, ablation, gamma sweep).
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import jax.numpy as jnp

from .config import Config
from .dynamics import make_simulate_eval, split_traj
from .params import assemble_full, make_frozen
from .preprocess import make_S0


def classify(image: np.ndarray, state, cfg: Config) -> dict:
    """Run a forward pass and return the predicted class plus diagnostics."""
    S0, mask = make_S0(image, D=state.D, tau=cfg.tau, n_max=cfg.n_max)
    w, mu, sigma = assemble_full(state.params, *state.frozen)
    traj = state.simulate_eval(S0, w, mu, sigma)
    q_T = traj[-1, :, :state.D]
    p_T = traj[-1, :, state.D:2 * state.D]
    com = jnp.sum(q_T * mask[:, None], axis=0) / jnp.maximum(jnp.sum(mask), 1)
    d_O = float(jnp.linalg.norm(com - state.q_star_O))
    d_X = float(jnp.linalg.norm(com - state.q_star_X))
    pred = "O" if d_O < d_X else "X"
    p_norm_mean = float(jnp.sum(jnp.linalg.norm(p_T, axis=-1) * mask)
                        / jnp.maximum(jnp.sum(mask), 1))
    return {
        "pred": pred, "d_O": d_O, "d_X": d_X,
        "com": np.asarray(com), "p_norm_mean": p_norm_mean,
        "traj": np.asarray(traj), "mask": np.asarray(mask),
    }


def accuracy(images: List[np.ndarray], labels: List[str],
             state, cfg: Config) -> float:
    correct = 0
    for img, lab in zip(images, labels):
        r = classify(img, state, cfg)
        if r["pred"] == lab:
            correct += 1
    return correct / len(images)


def noise_sweep(canonical: np.ndarray, true_label: str,
                state, cfg: Config,
                levels: List[int] = None, trials: int = 5, seed: int = 0) -> dict:
    """Flip 'n_flip' random pixels and record accuracy."""
    if levels is None:
        levels = list(range(0, 11))
    rng = np.random.RandomState(seed)
    out = {"levels": levels, "acc": []}
    for n_flip in levels:
        hits = 0
        for _ in range(trials):
            img = canonical.copy()
            if n_flip > 0:
                idx = rng.choice(img.size, size=n_flip, replace=False)
                rs, cs = np.unravel_index(idx, img.shape)
                img[rs, cs] = 1.0 - img[rs, cs]
            try:
                r = classify(img, state, cfg)
            except ValueError:
                # No particles above tau (rare); skip this trial.
                continue
            if r["pred"] == true_label:
                hits += 1
        out["acc"].append(hits / trials if trials else 0.0)
    return out


def shift_sweep(canonical: np.ndarray, true_label: str,
                state, cfg: Config, dxs=range(-2, 3), dys=range(-2, 3)) -> dict:
    """Translate the image and record accuracy on each (dx, dy)."""
    dxs, dys = list(dxs), list(dys)
    grid = np.zeros((len(dys), len(dxs)), dtype=float)
    for ix, dx in enumerate(dxs):
        for iy, dy in enumerate(dys):
            shifted = np.zeros_like(canonical)
            R, C = canonical.shape
            for r in range(R):
                for c in range(C):
                    rr, cc = r + dy, c + dx
                    if 0 <= rr < R and 0 <= cc < C:
                        shifted[rr, cc] = canonical[r, c]
            try:
                r = classify(shifted, state, cfg)
                grid[iy, ix] = 1.0 if r["pred"] == true_label else 0.0
            except ValueError:
                grid[iy, ix] = np.nan
    return {"dxs": dxs, "dys": dys, "grid": grid,
            "acc": float(np.nanmean(grid))}


def gamma_sweep(canonical_O: np.ndarray, canonical_X: np.ndarray,
                state, cfg: Config, gammas: List[float] = None) -> dict:
    """Re-simulate the canonical images with different gamma and record acc."""
    if gammas is None:
        gammas = [0.5, 1.0, 1.5, 2.0, 3.0]
    out = {"gammas": gammas, "acc": []}
    for g in gammas:
        sim_eval = make_simulate_eval(state.D, g, cfg.dt, cfg.n_steps)
        prev = state.simulate_eval
        state.simulate_eval = sim_eval
        try:
            rO = classify(canonical_O, state, cfg)
            rX = classify(canonical_X, state, cfg)
            ok = int(rO["pred"] == "O") + int(rX["pred"] == "X")
            out["acc"].append(ok / 2.0)
        finally:
            state.simulate_eval = prev
    return out


def ablation_zero_out(state, k_indices_to_zero: List[int]):
    """Return a 'shadow' params dict with selected w_k zeroed out.

    k_indices are into the *learnable* w array, NOT the full (frozen+learn) one.
    """
    w = np.asarray(state.params["w"]).copy()
    for k in k_indices_to_zero:
        if 0 <= k < len(w):
            w[k] = 0.0
    return {
        "w": jnp.asarray(w),
        "mu": state.params["mu"],
        "sigma_raw": state.params["sigma_raw"],
    }


def ablation_study(canonical_O: np.ndarray, canonical_X: np.ndarray,
                   state, cfg: Config) -> dict:
    """Standard 4-variant ablation: full / no stones / no free / attractors only."""
    K_learn = state.K_learn
    n_stones = cfg.n_stones
    stones_idx = list(range(0, n_stones))
    free_idx = list(range(n_stones, K_learn))
    variants = {
        "full":       [],
        "no_stones":  stones_idx,
        "no_free":    free_idx,
        "attractors": stones_idx + free_idx,
    }

    out = {}
    original_params = state.params
    for name, zero in variants.items():
        state.params = ablation_zero_out(state, zero)
        rO = classify(canonical_O, state, cfg)
        rX = classify(canonical_X, state, cfg)
        out[name] = {
            "pred_O": rO["pred"], "pred_X": rX["pred"],
            "d_O_to_O*": rO["d_O"], "d_X_to_X*": rX["d_X"],
            "acc": (int(rO["pred"] == "O") + int(rX["pred"] == "X")) / 2.0,
        }
    state.params = original_params
    return out
