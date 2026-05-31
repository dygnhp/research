"""
Forward-only N-class evaluation.

`classify(image, state, cfg)` runs the simulator once and assigns the
image to the class whose attractor is closest to the masked center of
mass at time T.  The rest of the file is convenience: accuracy on a
collection of images, plus three robustness sweeps (noise / shift /
gamma) and a standard ablation study.

All functions in this module are forward-only and never touch the
optimizer state, so they are safe to use on a frozen trained model.
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np
import jax.numpy as jnp

from .config import Config
from .dynamics import make_simulate_eval
from .params import assemble_full
from .preprocess import make_S0


def classify(image: np.ndarray, state, cfg: Config) -> dict:
    """Predict the class label for a single image."""
    S0, mask = make_S0(image, D=state.D, tau=cfg.tau, n_max=cfg.n_max)
    w, mu, sigma = assemble_full(state.params, *state.frozen)
    traj = state.simulate_eval(S0, w, mu, sigma)
    q_T = traj[-1, :, :state.D]
    p_T = traj[-1, :, state.D:2 * state.D]
    com = jnp.sum(q_T * mask[:, None], axis=0) / jnp.maximum(jnp.sum(mask), 1)
    com_np = np.asarray(com)

    distances = {}
    for c, lab in enumerate(state.class_labels):
        d = float(jnp.linalg.norm(com - state.q_stars[c]))
        distances[lab] = d
    pred = min(distances, key=distances.get)

    p_norm_mean = float(jnp.sum(jnp.linalg.norm(p_T, axis=-1) * mask)
                        / jnp.maximum(jnp.sum(mask), 1))
    return {
        "pred": pred,
        "distances": distances,
        "com": com_np,
        "p_norm_mean": p_norm_mean,
        "traj": np.asarray(traj),
        "mask": np.asarray(mask),
    }


def accuracy(images: List[np.ndarray], labels: List[str],
             state, cfg: Config) -> float:
    correct = 0
    total = 0
    for img, lab in zip(images, labels):
        try:
            r = classify(img, state, cfg)
        except ValueError:
            continue
        total += 1
        if r["pred"] == lab:
            correct += 1
    return correct / max(total, 1)


def confusion_matrix(images: List[np.ndarray], labels: List[str],
                     state, cfg: Config) -> Dict:
    """Return a confusion matrix and per-class accuracy."""
    class_set = state.class_labels
    cm = np.zeros((len(class_set), len(class_set)), dtype=int)
    idx = {lab: i for i, lab in enumerate(class_set)}
    for img, true_lab in zip(images, labels):
        try:
            r = classify(img, state, cfg)
        except ValueError:
            continue
        cm[idx[true_lab], idx[r["pred"]]] += 1
    per_class = []
    for i in range(len(class_set)):
        total = cm[i].sum()
        per_class.append(cm[i, i] / max(total, 1))
    overall = cm.trace() / max(cm.sum(), 1)
    return {
        "labels": class_set,
        "matrix": cm,
        "per_class_acc": per_class,
        "overall_acc": float(overall),
    }


def noise_sweep(canonical: np.ndarray, true_label: str,
                state, cfg: Config,
                levels: List[int] = None, trials: int = 5,
                seed: int = 0) -> dict:
    if levels is None:
        # noise levels scale roughly with image area
        H, W = canonical.shape
        max_flips = max(10, H * W // 6)
        levels = list(np.linspace(0, max_flips, 11, dtype=int))
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
                continue
            if r["pred"] == true_label:
                hits += 1
        out["acc"].append(hits / trials if trials else 0.0)
    return out


def shift_sweep(canonical: np.ndarray, true_label: str,
                state, cfg: Config, max_shift: int = 2) -> dict:
    dxs = list(range(-max_shift, max_shift + 1))
    dys = list(range(-max_shift, max_shift + 1))
    grid = np.zeros((len(dys), len(dxs)), dtype=float)
    H, W = canonical.shape
    for ix, dx in enumerate(dxs):
        for iy, dy in enumerate(dys):
            shifted = np.zeros_like(canonical)
            for r in range(H):
                for c in range(W):
                    rr, cc = r + dy, c + dx
                    if 0 <= rr < H and 0 <= cc < W:
                        shifted[rr, cc] = canonical[r, c]
            try:
                rr = classify(shifted, state, cfg)
                grid[iy, ix] = 1.0 if rr["pred"] == true_label else 0.0
            except ValueError:
                grid[iy, ix] = np.nan
    return {"dxs": dxs, "dys": dys, "grid": grid,
            "acc": float(np.nanmean(grid))}


def gamma_sweep(canonical_per_class: Dict[str, np.ndarray],
                state, cfg: Config, gammas: List[float] = None) -> dict:
    """Re-simulate the canonical images for each class with several gammas."""
    if gammas is None:
        gammas = [0.5, 1.0, 1.5, 2.0, 3.0]
    out = {"gammas": gammas, "acc": []}
    prev_sim = state.simulate_eval
    try:
        for g in gammas:
            state.simulate_eval = make_simulate_eval(
                state.D, g, cfg.dt, cfg.n_steps,
                cfg.sigma_min, cfg.sigma_max)
            correct = 0
            total = 0
            for lab, img in canonical_per_class.items():
                r = classify(img, state, cfg)
                total += 1
                if r["pred"] == lab:
                    correct += 1
            out["acc"].append(correct / max(total, 1))
    finally:
        # Always restore the original simulator, even if classify raised
        # mid-sweep, so the caller's state object is left consistent.
        state.simulate_eval = prev_sim
    return out


def ablation_zero_out(state, k_indices_to_zero: List[int]):
    """Build a params dict with the listed learnable indices zeroed."""
    w = np.asarray(state.params["w"]).copy()
    for k in k_indices_to_zero:
        if 0 <= k < len(w):
            w[k] = 0.0
    out = {
        "w": jnp.asarray(w),
        "mu": state.params["mu"],
        "sigma_raw": state.params["sigma_raw"],
    }
    if "attractor_sigma_raw" in state.params:
        out["attractor_sigma_raw"] = state.params["attractor_sigma_raw"]
    return out


def ablation_study(canonical_per_class: Dict[str, np.ndarray],
                   state, cfg: Config) -> dict:
    """Standard 4-variant ablation:
        full        / no_stones / no_free / attractors_only
    """
    K_learn = state.K_learn
    n_classes = cfg.n_classes
    stones_idx = list(range(0, n_classes))      # first C learnable are stones
    free_idx = list(range(n_classes, K_learn))  # the rest are free RBFs
    variants = {
        "full":              [],
        "no_stones":         stones_idx,
        "no_free":           free_idx,
        "attractors_only":   stones_idx + free_idx,
    }
    out = {}
    original = state.params
    try:
        for name, zero in variants.items():
            state.params = ablation_zero_out(state, zero)
            preds, correct = {}, 0
            for lab, img in canonical_per_class.items():
                try:
                    r = classify(img, state, cfg)
                    preds[lab] = r["pred"]
                    if r["pred"] == lab:
                        correct += 1
                except ValueError:
                    preds[lab] = "<error>"
            out[name] = {
                "predictions": preds,
                "acc": correct / max(len(canonical_per_class), 1),
            }
    finally:
        # Restore the original params unconditionally so that an unexpected
        # exception in classify does not leave the caller's state holding
        # the ablated parameters.
        state.params = original
    return out
