"""
Configuration for the Contact Hamiltonian Machine (CHM).

A single dataclass-like Config object holds every hyperparameter so that
training, evaluation, and growth share the same source of truth.

All numeric defaults follow the MD specification (Sections 3-9).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple
import numpy as np


@dataclass
class Config:
    # --------------------------------------------------------------
    # Physics (Sections 3-4)
    # --------------------------------------------------------------
    gamma: float = 1.5            # contact dissipation coefficient
    t_final: float = 10.0         # total simulation horizon
    dt: float = 0.05              # RK4 step size
    tau: float = 0.5              # pixel-intensity threshold for particle filtering
    n_max: int = 64               # padded particle slot count (JIT-stable shape)

    # --------------------------------------------------------------
    # Dimension and basis count (Section 7, 8)
    # --------------------------------------------------------------
    D_init: int = 3               # starting embedding dimension
    K_init: int = 16              # starting RBF count
    n_frozen: int = 2             # frozen attractors (O / X)
    n_stones: int = 2             # stepping stones (data-proximal)
    K_grow: int = 4               # RBFs added per K-growth event
    D_max: int = 8                # cap on dimension growth (unbounded conceptually,
                                  # but kept finite for JIT compile budget)
    K_max: int = 64               # cap on basis count

    # --------------------------------------------------------------
    # Training (Section 6)
    # --------------------------------------------------------------
    n_epochs: int = 3000
    peak_lr: float = 5e-3
    end_lr: float = 1e-5
    warmup_steps: int = 100
    grad_clip: float = 1.0
    lambda_p: float = 0.1          # momentum-penalty weight in the loss

    # --------------------------------------------------------------
    # Growth triggers (Section 7, 8)
    # --------------------------------------------------------------
    plateau_window: int = 100      # epochs averaged when checking plateau
    plateau_threshold: float = 0.01
    min_epochs_before_grow: int = 200
    cooldown_after_grow: int = 100
    K_grows_before_D: int = 3      # consecutive K-grows before trying D-grow

    # --------------------------------------------------------------
    # Convergence gates (Section 5, 9)
    # --------------------------------------------------------------
    eps_q_thresh: float = 2.0
    eps_p_thresh: float = 0.5
    phase_R2_thresh: float = 0.90

    # --------------------------------------------------------------
    # Dataset (Section 2)
    # --------------------------------------------------------------
    n_train_per_class: int = 50
    dataset_seed: int = 42

    # --------------------------------------------------------------
    # Attractor coordinates in the (x, y) plane (Section 5)
    # --------------------------------------------------------------
    q_star_O_xy: Tuple[float, float] = (8.0, 8.0)
    q_star_X_xy: Tuple[float, float] = (-8.0, -8.0)

    # --------------------------------------------------------------
    # I/O
    # --------------------------------------------------------------
    output_dir: str = "kanzen_runs"
    log_every: int = 20
    save_every: int = 500

    @property
    def n_steps(self) -> int:
        return int(self.t_final / self.dt)

    def q_star(self, label: str, D: int) -> np.ndarray:
        """Build the attractor target in D dimensions.

        The (x, y) coordinates are fixed by config; remaining dimensions are 0.
        For D >= 3 the z-channel is set to the class-typical connectivity value
        (0.88 for O, 0.12 for X) so the attractor lives on the class's z-plane.
        """
        base_xy = self.q_star_O_xy if label == "O" else self.q_star_X_xy
        out = np.zeros(D, dtype=np.float32)
        out[0] = base_xy[0]
        out[1] = base_xy[1]
        if D >= 3:
            out[2] = 0.88 if label == "O" else 0.12
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
