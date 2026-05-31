"""
kanzen -- Contact Hamiltonian Machine reference implementation.

Three datasets are bundled:

    OX_8     8x8   O / X
    ABC_16   16x16 A / B / C
    abcd_32  32x32 a / b / c / d

Select a dataset via Config.with_dataset(name).  See ALGORITHM.md for the
mechanism and EXPERIMENTS.md for recommended hyperparameters.
"""

# ---------------------------------------------------------------------------
# GPU / device environment (PART 4-1).
#
# These XLA flags must be set BEFORE jax is imported anywhere, so they live
# at the very top of the package __init__, ahead of every submodule import
# (the imports below transitively pull in jax).  setdefault is used so a
# user who exported their own value (or a launcher script that set one
# earlier) is never overridden.
#
#   XLA_PYTHON_CLIENT_PREALLOCATE=false
#       Allocate GPU memory on demand instead of grabbing ~75% up front,
#       so several small CHM runs can share one card.
#   XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
#       Cap JAX at 80% of GPU memory, leaving headroom for the driver and
#       any concurrent process.
# ---------------------------------------------------------------------------
import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")


def report_device():
    """Print and return the JAX execution backend ('GPU' / 'CPU' / 'TPU').

    Imports jax lazily so that merely importing the kanzen package does not
    force backend initialization until a caller actually asks.  The return
    value is the platform string of the first device, which PART 3/runner
    surfaces to the user.
    """
    import jax
    devices = jax.devices()
    kind = devices[0].platform.upper()   # 'GPU' or 'CPU'
    print(f"[device] JAX 실행 디바이스: {kind} ({len(devices)}개)")
    for d in devices:
        print(f"[device]   {d}")
    return kind


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
from .gallery import make_dataset_gallery
from .terrain_evolution import make_terrain_evolution_3d

__all__ = [
    "report_device",
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
    "make_dataset_gallery", "make_terrain_evolution_3d",
]
