# CHM — FINAL (experiment-ready)

The cleaned, experiment-ready Contact Hamiltonian Machine (N-class) package.
This is the version to **adjust parameters, run experiments from the CLI, and
produce result reports**. The pre-change baseline lives in `../LEGACY/`.

```
research_main/        <- run commands from HERE (the parent of FINAL/)
  FINAL/
    *.py              the package (flat) -> import name is `FINAL`
    README.md         this file
```

`FINAL/` is itself the Python package: the modules live directly under it and
use relative imports, so invoke them as `python -m FINAL.<cmd>` **from the
parent directory** (`research_main/`). Do not `cd` into FINAL/ to run — the
relative imports need the `FINAL` package context.

---

## Quick start — the experiment runner

```bash
cd research_main        # the directory that CONTAINS FINAL/

# estimate only (live calibration on this machine, no training):
python -m FINAL.runner --dataset OX --epochs 3000 --init-k 16 --init-d 3 --dry-run

# run an experiment (gallery -> train -> 3D terrain evolution -> eval -> report):
python -m FINAL.runner --dataset OX --epochs 3000 --init-k 16 --init-d 3 --yes
```

Outputs land in `research/experiment/exp_NNN/` (relative to where you run),
containing: `results.md`, `dataset_gallery.png`, `terrain_evolution_3d.png`,
`params.npz`, `config.json`, `history.json`, `growth_log.json`.

### Runner flags

| flag | meaning | default |
|---|---|---|
| `--dataset {OX,ABC,abcd}` | OX_8 / ABC_16 / abcd_32 | OX |
| `--epochs N` | training budget | 2000 |
| `--init-k K` | initial RBF count (>= 2*classes) | 16 |
| `--init-d D` | initial dimension (>= 2) | 3 |
| `--seed S` | reproducibility (fix it for A/B!) | 42 |
| `--attractor {default,improved}` | attractor layout | default |
| `--learn-sigma / --no-learn-sigma` | learn attractor sigma | learn |
| `--device {auto,gpu,cpu}` | backend (Windows = CPU; GPU via WSL2) | auto |
| `--phase-gallery` | after the run, render per-particle phase-space galleries | off |
| `--phase-max-particles N` | particles per class for `--phase-gallery` (0 = all) | 12 |
| `--dry-run` / `--yes` | estimate only / skip confirm | — |

### Per-particle phase-space gallery

For each class reference, render one gallery per particle with a `(q_d, p_d)`
phase-plane per dimension -- the position-vs-momentum trajectory over time
(line + equal-time sample points, colored by time; the dotted line is `p=0`
and the dashed line is the attractor target `q*_d`). The contact dynamics is
dissipative, so each trajectory settles toward `(q*_d, 0)`.

Generate during a run, or standalone on a saved experiment:

```bash
# during a run:
python -m FINAL.runner --dataset OX --epochs 3000 --phase-gallery --yes

# standalone, from any saved exp_NNN/ (params.npz + config.json):
python -m FINAL.phase_gallery --exp research/experiment/exp_001 \
       --class O --max-particles 12
```

Output: `<exp>/phase_space/<label>_pNN.png`. NB: a class reference can have
100+ particles (abcd), so `--phase-max-particles` caps the count.

---

## What changed vs LEGACY (the "B + sigma" integration)

Two coupled improvements target the "soft basin routing" issue (attractors
sitting 4-8 sigma from the particle cloud, so their gradient never reaches the
data domain and the CoM cannot settle on the true attractor):

1. **Attractor repositioning (`--attractor improved`).** Places each class
   attractor symmetric about the *data center* and within ~1.5-2 sigma of the
   particle cloud (OX: X moved to (-1,-1); ABC/abcd: polygon radius halved).
   `mu` (position) and `w` (depth) stay frozen.
2. **Learnable attractor sigma (`--learn-sigma`, on by default).** The
   attractor influence radius is now a trained parameter
   (`params["attractor_sigma_raw"]`, shape `(C,)`) instead of a hand-set
   constant, with a weak L2 pull toward its init and a slower learning rate.
   This removes the manual sigma tuning that contradicted CHM's
   self-interpretability claim.

Verified A/B (OX_8, 400 epochs, no growth, same seed):

| condition | eps_q O | eps_q X |
|---|---|---|
| default, frozen sigma (= LEGACY) | 6.7 | 14.2 |
| default, learned sigma | 6.6 | 14.2 |
| **improved, learned sigma** | **5.2** | **5.2** |

Learning sigma alone cannot fix a far `mu` (X stays 14.2); the repositioning
is what restores symmetry and reach. The two compose.

### A/B experiment recipe (fix the seed!)

```bash
python -m FINAL.runner --dataset OX --epochs 3000 --attractor default  --yes
python -m FINAL.runner --dataset OX --epochs 3000 --attractor improved --yes
```

### Reproducing LEGACY behaviour exactly

```bash
python -m FINAL.runner --dataset OX --attractor default --no-learn-sigma ...
```
(`--attractor default --no-learn-sigma` freezes the attractor sigma at init
and keeps the original layout, matching the pre-change frozen-sigma model.)

---

## Physics core is unchanged

`contact_rhs` (dq/dp/dz), the RBF potential and its analytic gradient, the
loss structure (position + lambda_p * momentum), the phase-volume R^2, and the
checkpointed BPTT are all untouched. Only attractor *placement* and the
*parameterization* of attractor sigma changed; the landscape is still a sum of
readable Gaussians (white-box preserved).
