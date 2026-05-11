"""
Training loop with autonomous K / D growth.

Outer loop structure (Section 7-9):

    while not settled and epoch < n_epochs:
        sample (img_O, img_X) from dataset
        compute loss, grad
        apply Adam update with grad clip
        every log_every epochs:
            evaluate canonical eps_q, eps_p, R^2
        if plateau and beyond min_epochs and cooldown elapsed:
            attempt grow_K   (until K_max)
            if grow_K has been attempted K_grows_before_D times in a row
                with no improvement -> grow_D (until D_max)
            rebuild simulator, optax state, frozen block at new shape
        if all three gates pass:
            settled = True

Shape changes during training force JAX to retrace and recompile.  The
training step is wrapped in a 'TrainerState' that we rebuild on every
growth event so the JIT cache only ever holds the *current* shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax

from .config import Config
from .data import generate_dataset, O_CANONICAL, X_CANONICAL
from .preprocess import make_S0
from .dynamics import make_simulate_diff, make_simulate_eval, split_traj
from .params import make_frozen, make_learnable, assemble_full
from .loss import loss_two_class, forward_and_metrics
from .invariants import epsilon_q, epsilon_p, phase_volume_R2
from .growth import PlateauDetector, grow_K, grow_D, grow_frozen_D


# ---------------------------------------------------------------------------
# Trainer state -- rebuilt on every shape change
# ---------------------------------------------------------------------------
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
    grad_fn: callable                                # value_and_grad of the loss
    q_star_O: jnp.ndarray
    q_star_X: jnp.ndarray


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
    """Build a TrainerState for current (D, K_learn).  If prev_params is
    given, warm-start from it; otherwise initialize from scratch."""
    frozen = make_frozen(cfg, D)
    if prev_params is None:
        params = make_learnable(cfg, D, rng_seed=cfg.dataset_seed)
    else:
        params = prev_params

    sim_diff = make_simulate_diff(D, cfg.gamma, cfg.dt, cfg.n_steps)
    sim_eval = make_simulate_eval(D, cfg.gamma, cfg.dt, cfg.n_steps)
    optimizer = _make_optimizer(cfg)
    opt_state = optimizer.init(params)

    q_star_O = jnp.asarray(cfg.q_star("O", D))
    q_star_X = jnp.asarray(cfg.q_star("X", D))

    def loss_fn(p, S0_O, mask_O, S0_X, mask_X):
        return loss_two_class(
            sim_diff, p, frozen, D,
            S0_O, mask_O, S0_X, mask_X,
            q_star_O, q_star_X, cfg.lambda_p,
        )

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    grad_fn = jax.jit(grad_fn)

    return TrainerState(
        D=D, K_learn=int(params["w"].shape[0]),
        params=params, frozen=frozen,
        simulate_diff=sim_diff, simulate_eval=sim_eval,
        optimizer=optimizer, opt_state=opt_state,
        grad_fn=grad_fn,
        q_star_O=q_star_O, q_star_X=q_star_X,
    )


# ---------------------------------------------------------------------------
# Canonical-image diagnostics (run every log_every epochs)
# ---------------------------------------------------------------------------
def _diagnostics(state: TrainerState, cfg: Config):
    S0_O, mask_O = make_S0(O_CANONICAL, D=state.D, tau=cfg.tau, n_max=cfg.n_max)
    S0_X, mask_X = make_S0(X_CANONICAL, D=state.D, tau=cfg.tau, n_max=cfg.n_max)
    w, mu, sigma = assemble_full(state.params, *state.frozen)

    traj_O = state.simulate_eval(S0_O, w, mu, sigma)
    traj_X = state.simulate_eval(S0_X, w, mu, sigma)
    qO_T = traj_O[-1, :, :state.D]
    pO_T = traj_O[-1, :, state.D:2 * state.D]
    qX_T = traj_X[-1, :, :state.D]
    pX_T = traj_X[-1, :, state.D:2 * state.D]

    com_O = jnp.sum(qO_T * mask_O[:, None], 0) / jnp.maximum(jnp.sum(mask_O), 1)
    com_X = jnp.sum(qX_T * mask_X[:, None], 0) / jnp.maximum(jnp.sum(mask_X), 1)

    eq_O = epsilon_q(com_O, state.q_star_O)
    eq_X = epsilon_q(com_X, state.q_star_X)
    ep_O = epsilon_p(pO_T, mask_O)
    ep_X = epsilon_p(pX_T, mask_X)

    R2_O = phase_volume_R2(traj_O, mask_O, state.D, cfg.gamma, cfg.dt)
    R2_X = phase_volume_R2(traj_X, mask_X, state.D, cfg.gamma, cfg.dt)

    return {
        "eps_q_O": eq_O, "eps_q_X": eq_X,
        "eps_p_O": ep_O, "eps_p_X": ep_X,
        "R2_O": R2_O, "R2_X": R2_X,
        "com_O": np.asarray(com_O),
        "com_X": np.asarray(com_X),
        "traj_O": traj_O, "traj_X": traj_X,
        "mask_O": np.asarray(mask_O), "mask_X": np.asarray(mask_X),
    }


def _settled(diag: dict, cfg: Config) -> bool:
    return (diag["eps_q_O"] < cfg.eps_q_thresh
            and diag["eps_q_X"] < cfg.eps_q_thresh
            and diag["eps_p_O"] < cfg.eps_p_thresh
            and diag["eps_p_X"] < cfg.eps_p_thresh
            and diag["R2_O"] >= cfg.phase_R2_thresh
            and diag["R2_X"] >= cfg.phase_R2_thresh)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
def train(cfg: Config, verbose: bool = True) -> dict:
    """Run the full training loop with autonomous growth.

    Returns a dict with the final trainer state, training history,
    and growth log.
    """
    rng = np.random.RandomState(cfg.dataset_seed)
    dataset = generate_dataset(cfg.n_train_per_class, cfg.dataset_seed)
    O_images = dataset["O_images"]
    X_images = dataset["X_images"]

    state = _build_state(cfg, D=cfg.D_init)
    detector = PlateauDetector(cfg.plateau_window, cfg.plateau_threshold)

    history = {"loss": [], "epoch": [], "diag": [], "events": []}
    growth_log: List[dict] = []

    last_growth_epoch = -10 ** 9
    consecutive_K_grows = 0

    for epoch in range(cfg.n_epochs):
        # ---- sample one O and one X per epoch (stochastic dataset training) -----
        i = rng.randint(0, len(O_images))
        j = rng.randint(0, len(X_images))
        S0_O, mask_O = make_S0(O_images[i], D=state.D, tau=cfg.tau, n_max=cfg.n_max)
        S0_X, mask_X = make_S0(X_images[j], D=state.D, tau=cfg.tau, n_max=cfg.n_max)

        (loss_val, aux), grads = state.grad_fn(state.params, S0_O, mask_O, S0_X, mask_X)
        updates, state.opt_state = state.optimizer.update(
            grads, state.opt_state, state.params)
        state.params = optax.apply_updates(state.params, updates)

        loss_val_f = float(loss_val)
        detector.update(loss_val_f)
        history["loss"].append(loss_val_f)
        history["epoch"].append(epoch)

        # ---- diagnostics + settlement check ------------------------------------
        if (epoch + 1) % cfg.log_every == 0 or epoch == cfg.n_epochs - 1:
            diag = _diagnostics(state, cfg)
            diag_log = {k: float(v) for k, v in diag.items()
                        if isinstance(v, (int, float)) or (isinstance(v, np.floating))
                        or (hasattr(v, 'shape') and v.shape == ())}
            history["diag"].append({"epoch": epoch, **diag_log})
            if verbose:
                print(f"[ep {epoch:5d}] loss={loss_val_f:.3f} "
                      f"D={state.D} K_learn={state.K_learn} "
                      f"eq_O={diag['eps_q_O']:.2f} eq_X={diag['eps_q_X']:.2f} "
                      f"ep_O={diag['eps_p_O']:.2f} ep_X={diag['eps_p_X']:.2f} "
                      f"R2_O={diag['R2_O']:.3f} R2_X={diag['R2_X']:.3f}")
            if _settled(diag, cfg):
                if verbose:
                    print(f"[settled at epoch {epoch}]")
                history["events"].append({"epoch": epoch, "event": "settled"})
                break

        # ---- growth trigger -----------------------------------------------------
        cooldown_ok = (epoch - last_growth_epoch) >= cfg.cooldown_after_grow
        beyond_min = epoch >= cfg.min_epochs_before_grow
        if cooldown_ok and beyond_min and detector.is_plateau():
            current_K_total = state.K_learn + cfg.n_frozen
            can_grow_K = (current_K_total + cfg.K_grow) <= cfg.K_max
            can_grow_D = (state.D + 1) <= cfg.D_max
            chose_D = (consecutive_K_grows >= cfg.K_grows_before_D
                       and can_grow_D)
            if (not chose_D) and can_grow_K:
                # ---- guided diagnostics for grow_K ----
                diag = _diagnostics(state, cfg)
                failing = []
                qstar_O_np = np.asarray(state.q_star_O)
                qstar_X_np = np.asarray(state.q_star_X)
                if diag["eps_q_O"] > cfg.eps_q_thresh:
                    failing.append((
                        np.asarray(diag["traj_O"][:, :, :state.D]),
                        diag["mask_O"], qstar_O_np))
                if diag["eps_q_X"] > cfg.eps_q_thresh:
                    failing.append((
                        np.asarray(diag["traj_X"][:, :, :state.D]),
                        diag["mask_X"], qstar_X_np))
                new_params, _ = grow_K(state.params, state.D,
                                       cfg.K_grow, failing)
                # rebuild full state (optimizer + jit cache) at new K
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
                # extend learnable params + sigma rescale; frozen is remade
                # at D_new inside _build_state via make_frozen.
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
            else:
                # nothing left to grow; let training continue at current shape
                pass

    return {
        "state": state,
        "history": history,
        "growth_log": growth_log,
        "config": cfg,
    }


def save_run(out_dir, run: dict) -> None:
    """Persist params, history, and growth log to disk."""
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
        # strip non-serializable trajectory arrays from diag entries before save
        history = run["history"]
        slim = {
            "loss": history["loss"],
            "epoch": history["epoch"],
            "diag": history["diag"],
            "events": history["events"],
        }
        json.dump(slim, f, indent=2)
    with open(os.path.join(out_dir, "growth_log.json"), "w") as f:
        json.dump(run["growth_log"], f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(run["config"].to_dict(), f, indent=2)
