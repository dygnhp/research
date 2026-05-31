"""
N-class training loop with autonomous K / D growth.

Outer loop structure:

    while not settled and epoch < n_epochs:
        sample one image per class
        stack S0 + mask along a leading class axis
        compute loss, grad
        Adam update with grad clip
        every log_every epochs:
            evaluate canonical eps_q, eps_p, R^2 for every class
        if plateau detected and beyond cooldown:
            attempt grow_K   (until K_total >= K_max)
            after K_grows_before_D consecutive K-grows: grow_D
            rebuild simulator and optimizer at the new shape
        if every class satisfies eps_q, eps_p, R^2 gates: settled

Shape changes force JAX to retrace and recompile, so we rebuild a
TrainerState object on every growth event.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax

from .config import Config
from .data import load_dataset
from .preprocess import make_S0
from .dynamics import make_simulate_diff, make_simulate_eval
from .params import make_frozen, make_learnable, assemble_full
from .loss import loss_batch
from .invariants import epsilon_q, epsilon_p, phase_volume_R2
from .growth import PlateauDetector, grow_K, grow_D


@dataclass
class TrainerState:
    D: int
    K_learn: int
    params: Dict[str, jnp.ndarray]
    frozen: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    simulate_diff: callable
    simulate_eval: callable
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    grad_fn: callable
    q_stars: jnp.ndarray     # (C, D)
    class_labels: List[str]


def _make_lr_schedule(cfg: Config) -> optax.Schedule:
    warmup = min(cfg.warmup_steps, max(1, cfg.n_epochs // 10))
    decay = max(1, cfg.n_epochs - warmup)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.peak_lr,
        warmup_steps=warmup,
        decay_steps=decay,
        end_value=cfg.end_lr,
    )


def _make_optimizer(cfg: Config) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=_make_lr_schedule(cfg), weight_decay=0.0),
    )


def _build_state(cfg: Config, D: int,
                 prev_params: Dict[str, jnp.ndarray] = None) -> TrainerState:
    frozen = make_frozen(cfg, D)
    if prev_params is None:
        params = make_learnable(cfg, D, rng_seed=cfg.dataset_seed)
    else:
        params = prev_params

    sim_diff = make_simulate_diff(D, cfg.gamma, cfg.dt, cfg.n_steps)
    sim_eval = make_simulate_eval(D, cfg.gamma, cfg.dt, cfg.n_steps)
    optimizer = _make_optimizer(cfg)
    opt_state = optimizer.init(params)
    q_stars = jnp.asarray(cfg.q_stars(D))   # (C, D)

    def loss_fn(p, S0_batch, mask_batch):
        return loss_batch(sim_diff, p, frozen, D,
                          S0_batch, mask_batch, q_stars, cfg.lambda_p)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    return TrainerState(
        D=D, K_learn=int(params["w"].shape[0]),
        params=params, frozen=frozen,
        simulate_diff=sim_diff, simulate_eval=sim_eval,
        optimizer=optimizer, opt_state=opt_state,
        grad_fn=grad_fn,
        q_stars=q_stars,
        class_labels=list(cfg.class_labels),
    )


def _sample_batch(images_by_label, class_labels, rng, D, cfg):
    """Sample one image per class, lift, and stack into a (C, N, 2D+1) batch."""
    S0_list, mask_list = [], []
    for label in class_labels:
        imgs = images_by_label[label]
        i = rng.randint(0, len(imgs))
        S0, mask = make_S0(imgs[i], D=D, tau=cfg.tau, n_max=cfg.n_max)
        S0_list.append(S0)
        mask_list.append(mask)
    return jnp.stack(S0_list, axis=0), jnp.stack(mask_list, axis=0)


def _diagnostics(state: TrainerState, cfg: Config):
    """Evaluate canonical eps_q, eps_p, R^2 for every class."""
    spec = cfg.dataset_spec
    w, mu, sigma = assemble_full(state.params, *state.frozen)
    per_class = {}
    eq_max = 0.0
    ep_max = 0.0
    R2_min = float("inf")
    for c, label in enumerate(state.class_labels):
        canon = spec.canonicals[label]
        S0, mask = make_S0(canon, D=state.D, tau=cfg.tau, n_max=cfg.n_max)
        traj = state.simulate_eval(S0, w, mu, sigma)
        q_T = traj[-1, :, :state.D]
        p_T = traj[-1, :, state.D:2 * state.D]
        com = jnp.sum(q_T * mask[:, None], 0) / jnp.maximum(jnp.sum(mask), 1)
        eq = epsilon_q(com, state.q_stars[c])
        ep = epsilon_p(p_T, mask)
        R2 = phase_volume_R2(traj, mask, state.D, cfg.gamma, cfg.dt)
        per_class[label] = {
            "eps_q": eq, "eps_p": ep, "R2": R2,
            "com": np.asarray(com),
            "traj": traj, "mask": np.asarray(mask),
        }
        eq_max = max(eq_max, eq)
        ep_max = max(ep_max, ep)
        if np.isfinite(R2):
            R2_min = min(R2_min, R2)
    return {
        "per_class": per_class,
        "eps_q_max": eq_max,
        "eps_p_max": ep_max,
        "R2_min": R2_min if np.isfinite(R2_min) else float("nan"),
    }


def _settled(diag: dict, cfg: Config) -> bool:
    return (diag["eps_q_max"] < cfg.eps_q_thresh
            and diag["eps_p_max"] < cfg.eps_p_thresh
            and (np.isnan(diag["R2_min"])
                 or diag["R2_min"] >= cfg.phase_R2_thresh))


def train(cfg: Config, verbose: bool = True) -> dict:
    """Run training with autonomous growth.  Returns final state + history."""
    rng = np.random.RandomState(cfg.dataset_seed)
    dataset = load_dataset(cfg.dataset, cfg.n_train_per_class, cfg.dataset_seed)
    images_by_label = dataset["images_by_label"]

    state = _build_state(cfg, D=cfg.D_init)
    detector = PlateauDetector(cfg.plateau_window, cfg.plateau_threshold)

    history = {"loss": [], "epoch": [], "diag": [], "events": []}
    growth_log: List[dict] = []
    last_growth_epoch = -10 ** 9
    consecutive_K_grows = 0

    for epoch in range(cfg.n_epochs):
        S0_batch, mask_batch = _sample_batch(
            images_by_label, state.class_labels, rng, state.D, cfg)

        (loss_val, aux), grads = state.grad_fn(state.params, S0_batch, mask_batch)
        updates, state.opt_state = state.optimizer.update(
            grads, state.opt_state, state.params)
        state.params = optax.apply_updates(state.params, updates)

        loss_f = float(loss_val)
        detector.update(loss_f)
        history["loss"].append(loss_f)
        history["epoch"].append(epoch)

        if (epoch + 1) % cfg.log_every == 0 or epoch == cfg.n_epochs - 1:
            diag = _diagnostics(state, cfg)
            entry = {
                "epoch": epoch,
                "eps_q_max": float(diag["eps_q_max"]),
                "eps_p_max": float(diag["eps_p_max"]),
                "R2_min":    float(diag["R2_min"]),
            }
            for lab, info in diag["per_class"].items():
                entry[f"eps_q_{lab}"] = float(info["eps_q"])
                entry[f"eps_p_{lab}"] = float(info["eps_p"])
                entry[f"R2_{lab}"]    = float(info["R2"])
            history["diag"].append(entry)
            if verbose:
                labels_str = ", ".join(
                    f"{lab}:eq={info['eps_q']:.2f}"
                    for lab, info in diag["per_class"].items())
                print(f"[ep {epoch:5d}] loss={loss_f:.3f} "
                      f"D={state.D} K_learn={state.K_learn} "
                      f"R2_min={diag['R2_min']:.3f} | {labels_str}")
            if _settled(diag, cfg):
                if verbose:
                    print(f"[settled at epoch {epoch}]")
                history["events"].append({"epoch": epoch, "event": "settled"})
                break

        # ---- growth trigger -----------------------------------------------
        cooldown_ok = (epoch - last_growth_epoch) >= cfg.cooldown_after_grow
        beyond_min = epoch >= cfg.min_epochs_before_grow
        if not (cooldown_ok and beyond_min and detector.is_plateau()):
            continue

        n_frozen = cfg.n_classes
        current_K_total = state.K_learn + n_frozen
        can_grow_K = (current_K_total + cfg.K_grow) <= cfg.K_max
        can_grow_D = (state.D + 1) <= cfg.D_max
        chose_D = (consecutive_K_grows >= cfg.K_grows_before_D and can_grow_D)
        if (not chose_D) and can_grow_K:
            diag = _diagnostics(state, cfg)
            failing = []
            for c, label in enumerate(state.class_labels):
                info = diag["per_class"][label]
                if info["eps_q"] > cfg.eps_q_thresh:
                    failing.append((
                        np.asarray(info["traj"][:, :, :state.D]),
                        info["mask"],
                        np.asarray(state.q_stars[c]),
                    ))
            new_params, _ = grow_K(state.params, state.D, cfg.K_grow, failing)
            state = _build_state(cfg, state.D, prev_params=new_params)
            last_growth_epoch = epoch
            consecutive_K_grows += 1
            growth_log.append({"epoch": epoch, "event": "grow_K",
                               "K_learn_after": state.K_learn,
                               "D": state.D})
            history["events"].append(growth_log[-1])
            detector.reset()
            if verbose:
                print(f"[ep {epoch}] >>> grow_K -> K_learn={state.K_learn}")
        elif chose_D:
            D_old = state.D
            D_new = D_old + 1
            new_params = grow_D(state.params, D_old, D_new)
            state = _build_state(cfg, D_new, prev_params=new_params)
            last_growth_epoch = epoch
            consecutive_K_grows = 0
            growth_log.append({"epoch": epoch, "event": "grow_D",
                               "D_after": state.D,
                               "K_learn": state.K_learn})
            history["events"].append(growth_log[-1])
            detector.reset()
            if verbose:
                print(f"[ep {epoch}] >>> grow_D -> D={state.D}")

    return {
        "state": state,
        "history": history,
        "growth_log": growth_log,
        "config": cfg,
    }


def save_run(out_dir, run: dict) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    state = run["state"]
    np.savez(
        os.path.join(out_dir, "params.npz"),
        w=np.asarray(state.params["w"]),
        mu=np.asarray(state.params["mu"]),
        sigma_raw=np.asarray(state.params["sigma_raw"]),
        D=state.D,
        K_learn=state.K_learn,
    )
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({
            "loss": run["history"]["loss"],
            "epoch": run["history"]["epoch"],
            "diag": run["history"]["diag"],
            "events": run["history"]["events"],
        }, f, indent=2)
    with open(os.path.join(out_dir, "growth_log.json"), "w") as f:
        json.dump(run["growth_log"], f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(run["config"].to_dict(), f, indent=2)
