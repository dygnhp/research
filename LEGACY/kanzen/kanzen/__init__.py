"""
kanzen -- Contact Hamiltonian Machine reference implementation.

Three datasets are bundled:

    OX_8     8x8   O / X
    ABC_16   16x16 A / B / C
    abcd_32  32x32 a / b / c / d

Select a dataset via Config.with_dataset(name).  See ALGORITHM.md for the
mechanism and EXPERIMENTS.md for recommended hyperparameters.
"""

from .config import Config
from .data import (
    DATASETS, DatasetSpec, load_dataset, generate_dataset,
    O_CANONICAL, X_CANONICAL,
    A_CANONICAL_16, B_CANONICAL_16, C_CANONICAL_16,
    a_CANONICAL_32, b_CANONICAL_32, c_CANONICAL_32, d_CANONICAL_32,
)
from .preprocess import preprocess, make_S0
from .terrain import rbf_potential, rbf_gradient
from .dynamics import (contact_rhs, make_rk4_step,
                       make_simulate_diff, make_simulate_eval, split_traj)
from .params import make_frozen, make_learnable, assemble_full
from .loss import loss_batch, forward_and_metrics
from .invariants import epsilon_q, epsilon_p, phase_volume_R2
from .growth import PlateauDetector, grow_K, grow_D, grow_frozen_D
from .train import train, save_run, TrainerState
from .evaluate import (classify, accuracy, confusion_matrix,
                       noise_sweep, shift_sweep, gamma_sweep, ablation_study)
from .viz import summary_figure

__all__ = [
    "Config",
    "DATASETS", "DatasetSpec", "load_dataset", "generate_dataset",
    "O_CANONICAL", "X_CANONICAL",
    "A_CANONICAL_16", "B_CANONICAL_16", "C_CANONICAL_16",
    "a_CANONICAL_32", "b_CANONICAL_32", "c_CANONICAL_32", "d_CANONICAL_32",
    "preprocess", "make_S0",
    "rbf_potential", "rbf_gradient",
    "contact_rhs", "make_rk4_step",
    "make_simulate_diff", "make_simulate_eval", "split_traj",
    "make_frozen", "make_learnable", "assemble_full",
    "loss_batch", "forward_and_metrics",
    "epsilon_q", "epsilon_p", "phase_volume_R2",
    "PlateauDetector", "grow_K", "grow_D", "grow_frozen_D",
    "train", "save_run", "TrainerState",
    "classify", "accuracy", "confusion_matrix",
    "noise_sweep", "shift_sweep", "gamma_sweep", "ablation_study",
    "summary_figure",
]
