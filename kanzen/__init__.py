"""
kanzen -- Contact Hamiltonian Machine reference implementation.

A clean, end-to-end build of the spec:

    image -> particles
                 |
                 v
       contact Hamiltonian ODE on a learnable RBF terrain
                 |
                 v
      CoM at time T  ->  class
                 |
                 v
       loss on (eps_q, eps_p) -> grad -> Adam update
                 |
                 v
       autonomous K / D growth when the landscape is too poor

See ALGORITHM.md (next to this file) for the full derivation.
"""

from .config import Config
from .data import O_CANONICAL, X_CANONICAL, generate_dataset
from .preprocess import preprocess, make_S0
from .terrain import rbf_potential, rbf_gradient
from .dynamics import (contact_rhs, make_rk4_step,
                       make_simulate_diff, make_simulate_eval, split_traj)
from .params import make_frozen, make_learnable, assemble_full
from .loss import loss_two_class, forward_and_metrics
from .invariants import epsilon_q, epsilon_p, phase_volume_R2
from .growth import PlateauDetector, grow_K, grow_D, grow_frozen_D
from .train import train, save_run
from .evaluate import (classify, accuracy, noise_sweep, shift_sweep,
                       gamma_sweep, ablation_study)
from .viz import summary_figure

__all__ = [
    "Config",
    "O_CANONICAL", "X_CANONICAL", "generate_dataset",
    "preprocess", "make_S0",
    "rbf_potential", "rbf_gradient",
    "contact_rhs", "make_rk4_step",
    "make_simulate_diff", "make_simulate_eval", "split_traj",
    "make_frozen", "make_learnable", "assemble_full",
    "loss_two_class", "forward_and_metrics",
    "epsilon_q", "epsilon_p", "phase_volume_R2",
    "PlateauDetector", "grow_K", "grow_D", "grow_frozen_D",
    "train", "save_run",
    "classify", "accuracy", "noise_sweep", "shift_sweep",
    "gamma_sweep", "ablation_study",
    "summary_figure",
]
