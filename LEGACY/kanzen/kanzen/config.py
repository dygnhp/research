"""
Configuration for the Contact Hamiltonian Machine (CHM).

A single Config dataclass holds every hyperparameter so that training,
evaluation, and growth share the same source of truth.

The 'dataset' field selects one of the entries in data.DATASETS, which
in turn determines the image size, the class set, the attractor layout,
and the padded particle slot count.

Per-dataset training defaults (n_epochs, K_init, plateau_window, etc.)
are baked into _DATASET_DEFAULTS so each experiment starts from sensible
values; any of them can still be overridden when the Config is
constructed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Tuple, Dict, List
import numpy as np


# ---------------------------------------------------------------------------
# Per-dataset training defaults
# ---------------------------------------------------------------------------
_DATASET_DEFAULTS: Dict[str, Dict] = {
    "OX_8": {
        "K_init":           16,    # 2 frozen + 2 stones + 12 free
        "n_epochs":         3000,
        "peak_lr":          5e-3,
        "warmup_steps":     100,
        "plateau_window":   100,
        "min_epochs_before_grow": 200,
        "eps_q_thresh":     2.0,
        "eps_p_thresh":     0.5,
        "phase_R2_thresh":  0.90,
    },
    "ABC_16": {
        "K_init":           21,    # 3 frozen + 3 stones + 15 free
        "n_epochs":         5000,
        "peak_lr":          3e-3,
        "warmup_steps":     200,
        "plateau_window":   150,
        "min_epochs_before_grow": 400,
        "eps_q_thresh":     3.0,
        "eps_p_thresh":     0.6,
        "phase_R2_thresh":  0.85,
    },
    "abcd_32": {
        "K_init":           28,    # 4 frozen + 4 stones + 20 free
        "n_epochs":         8000,
        "peak_lr":          2e-3,
        "warmup_steps":     300,
        "plateau_window":   200,
        "min_epochs_before_grow": 600,
        "eps_q_thresh":     5.0,
        "eps_p_thresh":     0.8,
        "phase_R2_thresh":  0.80,
    },
}


@dataclass
class Config:
    # ---- Dataset selection -------------------------------------------------
    dataset: str = "OX_8"

    # ---- Physics -----------------------------------------------------------
    gamma: float = 1.5
    t_final: float = 10.0
    dt: float = 0.05

    # ---- Dimension and basis count ----------------------------------------
    D_init: int = 3
    K_init: int = 16
    K_grow: int = 4
    D_max: int = 8
    K_max: int = 64

    # ---- Training ----------------------------------------------------------
    n_epochs: int = 3000
    peak_lr: float = 5e-3
    end_lr: float = 1e-5
    warmup_steps: int = 100
    grad_clip: float = 1.0
    lambda_p: float = 0.1

    # ---- Growth triggers ---------------------------------------------------
    plateau_window: int = 100
    plateau_threshold: float = 0.01
    min_epochs_before_grow: int = 200
    cooldown_after_grow: int = 100
    K_grows_before_D: int = 3

    # ---- Convergence gates -------------------------------------------------
    eps_q_thresh: float = 2.0
    eps_p_thresh: float = 0.5
    phase_R2_thresh: float = 0.90

    # ---- Dataset sampling --------------------------------------------------
    n_train_per_class: int = 50
    dataset_seed: int = 42

    # ---- I/O ---------------------------------------------------------------
    output_dir: str = "kanzen_runs"
    log_every: int = 20
    save_every: int = 500

    # ---- internal: which keys were left at default vs explicitly set ------
    _explicit_keys: set = field(default_factory=set, repr=False)

    def __post_init__(self):
        """Apply per-dataset defaults for any field not explicitly overridden.

        We detect explicit overrides by comparing each field to its
        dataclass default at construction.  The 'with_dataset' helper below
        is the recommended way to construct a Config since it makes the
        intent explicit.
        """
        defaults = _DATASET_DEFAULTS.get(self.dataset, {})
        for key, val in defaults.items():
            if key in self._explicit_keys:
                continue
            if hasattr(self, key):
                setattr(self, key, val)

    @classmethod
    def with_dataset(cls, name: str, **overrides) -> "Config":
        """Construct a Config for the given dataset, with optional overrides.

        Example:
            cfg = Config.with_dataset("ABC_16", n_epochs=2000)
        """
        explicit = set(overrides.keys()) | {"dataset"}
        kwargs = dict(overrides)
        kwargs["dataset"] = name
        kwargs["_explicit_keys"] = explicit
        return cls(**kwargs)

    # ---- Convenience accessors (lazy-import to avoid circular dep) --------
    @property
    def dataset_spec(self):
        from .data import DATASETS
        return DATASETS[self.dataset]

    @property
    def n_classes(self) -> int:
        return self.dataset_spec.n_classes

    @property
    def class_labels(self) -> List[str]:
        return self.dataset_spec.class_labels

    @property
    def n_max(self) -> int:
        return self.dataset_spec.n_max

    @property
    def tau(self) -> float:
        return self.dataset_spec.tau

    @property
    def n_steps(self) -> int:
        return int(self.t_final / self.dt)

    # ---- attractor query --------------------------------------------------
    def q_star(self, label: str, D: int) -> np.ndarray:
        """Attractor position for a class label in D dimensions.

        Dimensions 0, 1 are filled with the (x, y) attractor; dimension 2
        (if present) gets the class-typical connectivity z; dimensions
        3+ default to 0.5 (mid-range).
        """
        spec = self.dataset_spec
        if label not in spec.attractor_positions:
            raise KeyError(f"Label '{label}' not in dataset {self.dataset}")
        x, y = spec.attractor_positions[label]
        out = np.zeros(D, dtype=np.float32)
        out[0] = x
        out[1] = y
        if D >= 3:
            out[2] = spec.attractor_z[label]
        if D >= 4:
            out[3] = 0.5
        return out

    def q_stars(self, D: int) -> np.ndarray:
        """Stack all class attractors into a (C, D) array."""
        return np.stack([self.q_star(lab, D) for lab in self.class_labels])

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_explicit_keys", None)
        return d
