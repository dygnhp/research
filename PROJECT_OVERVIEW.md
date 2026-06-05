# CHM Project — Overview for AI Agents

Onboarding doc for another Claude Code (or any agent) picking up this repo.
Read this first, then `FINAL/PIPELINE.md` (code data-flow) and
`EXPERIMENT_RESULTS.md` (all numeric results).

---

## 1. What this is

The **Contact Hamiltonian Machine (CHM)** is a *white-box* image classifier.
Instead of weight-matrix multiplications, it turns each image into a cloud of
physical particles, lets them move on a learnable RBF potential landscape under
a dissipative (contact-Hamiltonian) ODE, and assigns the class whose attractor
the particle ensemble settles toward.

```
image -> particles -> motion on a learned landscape -> settle near an attractor -> class
```

The whole forward path is one trajectory of an explicit ODE, so the decision is
inspectable (plot the landscape, the CoM path, or per-particle phase portraits).
The research goal is **interpretability by construction**, not accuracy SOTA.

Datasets (synthetic glyphs, generated parametrically):
`OX_8` (8x8, O/X), `ABC_16` (16x16, A/B/C), `abcd_32` (32x32, a/b/c/d).

---

## 2. Repository layout (READ THIS — there are traps)

```
research_main/
  FINAL/            <- CURRENT code. A flat Python package; import name `FINAL`.
                       Run as `python -m FINAL.<module>` FROM THE REPO ROOT.
                       Do NOT `cd FINAL` (relative imports need the FINAL pkg).
  LEGACY/           <- archived old project (pre-reorg). Contains an OLDER copy
                       of the CHM code under LEGACY/kanzen/. Do NOT edit for new
                       work. Reproduces the pre-"learnable sigma / improved
                       attractor" behaviour.
  control/          <- TensorFlow ANN baseline (control group). Independent of
                       FINAL's physics. `python -m control.ann_baseline`.
  research/          <- experiment OUTPUTS (gitignored: large 3D PNGs + npz).
                       Findings are committed in EXPERIMENT_RESULTS.md instead.
  EXPERIMENT_RESULTS.md   all numeric results (ablation, GPU bench, main exp, ANN)
  PROJECT_OVERVIEW.md     this file
  FINAL/PIPELINE.md       per-file roles + inter-file data flow + B/sigma wiring
  FINAL/README.md         runner CLI usage
```

**Traps an agent WILL hit:**
- **Two same-name packages.** `LEGACY/kanzen/` is an old CHM copy; `import kanzen`
  vs `import FINAL` can resolve to different code depending on cwd. Always use
  `FINAL` from the repo root for current work.
- **Native Windows JAX is CPU-only.** `pip install "jax[cuda12]"` does NOT work on
  Windows (Linux-only wheels). GPU is via **WSL2** (Ubuntu) only — see §4.
- **16 GB RAM machine.** Running WSL-GPU and Windows-CPU jobs *concurrently*
  spills to the HDD pagefile (D:) and everything stalls. Run heavy jobs one at a
  time.
- **git-bash path mangling.** `wsl -d Ubuntu -- bash /mnt/c/...` gets the `/mnt`
  path rewritten to `C:/Program Files/Git/mnt/...`. Put the path INSIDE a
  `bash -lc '...'` single-quoted string instead.
- **Korean console (cp949).** Symbols like `≈ → ═ × σ`(U+2248 etc.) can crash
  `print()` on the Windows console. Keep in-figure/console text ASCII-safe;
  Korean text itself is fine (cp949), exotic math symbols are not.

---

## 3. The CHM mechanism (just enough to be dangerous)

- `data.py` generates glyph images. `preprocess.py` lifts on-pixels (>tau) to
  particles in R^D: dim0=x(col), dim1=y(row-flip), dim2=z_connectivity,
  dim3=local_density, dim4+=0. Padded to `n_max`; a boolean `mask` marks real
  particles (dummies never count).
- `params.py` builds the RBF terrain in 3 blocks: **attractors** (one per class,
  mu/w FROZEN, **sigma LEARNED** — see below), **stepping stones** (learned),
  **free basis** (learned). `assemble_full` stacks them into `(w, mu, sigma)`.
- `terrain.py` = `V(q)=sum_k w_k exp(-||q-mu_k||^2/2 sigma_k^2)` and analytic grad.
- `dynamics.py` = contact eqns `dq=p; dp=-gradV-gamma*p; dz=||p||^2-H`, fixed-step
  RK4 in `lax.scan`, `jax.checkpoint` for memory-efficient BPTT.
- `loss.py` = masked CoM-to-attractor distance + lambda_p*residual-momentum +
  weak L2 on attractor sigma.
- `train.py` = epoch loop, `value_and_grad`, optax (with a separate slower route
  for attractor sigma), autonomous growth (grow_K / grow_D), terrain snapshots.
- `invariants.py` = eps_q (CoM-target dist), eps_p (residual speed), phase-volume
  R^2 — the "settled" gates.
- `evaluate.py` = forward-only classify (argmin CoM-to-attractor) + accuracy.
- viz: `gallery.py` (dataset grid), `terrain_evolution.py` (3D V(x,y) over epochs
  + leading reference panels), `phase_gallery.py` (per-particle (q_d,p_d)).

**Key design change made in this work (the "B + learnable sigma" integration):**
- Attractor **sigma** moved from frozen to a learned param (`attractor_sigma_raw`,
  shape `(C,)`), so the influence radius is learned, not hand-set. Attractor
  **mu (label anchor) and w (anti-trivial-solution) stay frozen.** Toggle:
  `Config.learn_attractor_sigma` / runner `--learn-sigma`.
- **"improved" attractor layout** (`--attractor improved`): data-center-symmetric
  placement within ~1.5-2 sigma of the cloud (vs the default origin-symmetric
  layout that left some classes 8 sigma away, gradient-dead). Opt-in via
  `Config.attractor_override`; the global `DATASETS` registry is never mutated.
- Both default to ON in FINAL; set `--attractor default --no-learn-sigma` to
  reproduce LEGACY behaviour.

---

## 4. Environments

- **Windows CPU** (default): `python -m FINAL.runner ...` from repo root. JAX 0.4.30 CPU.
- **WSL2 GPU** (RTX 4060): a venv at `~/chm_gpu` in Ubuntu has `jax[cuda12]`
  (jax 0.10.1) + optax + matplotlib + (tensorflow). Run:
  ```bash
  wsl -d Ubuntu -- bash -lc 'source ~/chm_gpu/bin/activate && \
    cd /mnt/c/Users/User/PycharmProjects/research_main && \
    python -m FINAL.runner --dataset abcd --epochs 8000 --device gpu --yes'
  ```
- **GPU helps only large problems**: WSL-GPU vs WSL-CPU ms/epoch — OX 62.9 vs 20.5
  (GPU 3x slower!), ABC 82.5 vs 119, abcd 129.7 vs 724 (GPU ~5.6x faster).
  Crossover ~ABC. Use CPU for OX/ABC, GPU for abcd. (WSL-CPU > Windows-CPU too.)
- **TensorFlow** (control group) is installed in Windows python (2.20). The ANN
  is CPU-forced.

---

## 5. HOW TO RUN EXPERIMENTS (methods — mandatory section)

### 5.1 The CHM experiment runner
`FINAL/runner.py` does: validate -> live 2-point time calibration -> dataset
gallery -> train (with autonomous growth) -> 3D terrain-evolution -> eval ->
`results.md`, optionally per-particle phase galleries. Outputs to
`<out-base>/exp_NNN/`.

```bash
# estimate only:
python -m FINAL.runner --dataset OX --epochs 3000 --init-k 16 --init-d 3 --dry-run
# full run:
python -m FINAL.runner --dataset OX --epochs 3000 --attractor improved --yes
```
Flags: `--dataset {OX,ABC,abcd}`, `--epochs`, `--init-k`, `--init-d`, `--seed`,
`--attractor {default,improved}`, `--learn-sigma/--no-learn-sigma`,
`--device {auto,gpu,cpu}`, `--phase-gallery [--phase-max-particles N]`,
`--out-base DIR`, `--yes`, `--dry-run`.

### 5.2 Standalone tools (on a saved exp_NNN/)
```bash
python -m FINAL.gallery   --dataset OX --n_per_class 50
python -m FINAL.phase_gallery --exp research/main_exp_2/exp_001 --class O --max-particles 12
```

### 5.3 The ANN control group (TensorFlow Sequential baseline)
Same images/seeds as CHM; `Flatten -> Dense(hidden, relu) -> Dense(C, softmax)`.
```bash
python -m control.ann_baseline --seeds 42,1,2,3,4 --hidden 32 --epochs 150
```
- `--seeds`: one ANN run per seed; controls the variant data (load_dataset),
  held-out set (seed+10000), and Keras weight init -> gives mean+-std.
- `--hidden`: hidden-layer width = model capacity / parameter count.
- Reports canonical / training-variant / held-out accuracy + param count, and a
  side-by-side vs CHM. FLOPs comparison is deferred.

### 5.4 Experimental methodology (the protocol used)
- **Configs compared (2x2 ablation):** {default, improved attractor} x
  {sigma frozen, sigma learned}. The "main experiments" then fix the best cell
  (improved + learned sigma).
- **Budgets (full):** OX 3000 / ABC 5000 / abcd 8000 epochs. Full budgets let
  `grow_D` fire — this was decisive (abcd jumped from ablation-budget 40% to
  full-budget 71%).
- **Seeds:** fix the seed per run; sweep multiple seeds (2-3 for CHM, 5 for the
  ANN) for mean+-std. Same seed across a comparison isolates the variable.
- **Devices:** OX/ABC on CPU, abcd on GPU (per §4).
- **Metrics:** canonical accuracy (the C reference glyphs), variant accuracy
  (over the training variants), eps_q / eps_p / R^2 settle gates, final D/K,
  growth events, learned attractor sigma, wall-time. ANN adds a held-out split.
- **Reproducibility:** runner saves `params.npz`, `config.json`, `history.json`,
  `growth_log.json` per run; re-simulate via `phase_gallery.load_state_from_exp`.

---

## 6. Results in one paragraph (full numbers in EXPERIMENT_RESULTS.md)

CHM (improved + learned sigma, full budget, multi-seed): **canonical 100% on all
three datasets**; variant **OX ~86% / ABC ~93% / abcd ~71%** with low seed
variance. The improved attractor layout halves eps_q vs default; sigma-learning
is a minor refinement. The **ANN control hits ~100% everywhere** (incl. abcd a/d
that CHM cannot separate) with near-zero variance — so the a/d distinction lives
in the pixels and CHM's limit is its CoM-of-near-identical-clouds representation,
not the data. **CHM trades accuracy + compute for interpretability.**

Open issues: the **phase-volume R^2 gate stays negative** (a deeper diagnostic
issue, separate from eps_q reachability). abcd a/c are the weak classes.
Learnable attractor **mu** was reviewed and **deferred** (trivial-solution risk;
doesn't fix near-identical inputs). FLOPs (compute) comparison is **deferred**.

---

## 7. Work timeline (dates from git)

| date (2026) | phase |
|---|---|
| 05-11 ~ 05-28 | (pre-session) prototype blocks; N-class CHM + "Phase A" attractor fix baseline (commits `bc0fcc6`, `94571c4`). |
| **06-01** | This work begins. A-bundle guideline fixes (grow_K image-size, evaluate try/finally, sigma-clip Config). **B (attractor repositioning) + learnable-sigma** integrated. Tooling: `report_device`/XLA env, `gallery`, `terrain_evolution` + snapshot hook, CLI `runner` with time calibration. **FINAL/LEGACY reorganization.** PR #2 (closed, superseded) -> **PR #3 "FINAL package" merged** (`ca3f314`). |
| **06-03** | CPU 2x2 ablation (12 cells); **GPU test** (set up WSL2 `jax[cuda12]`, benchmark); **1st main experiment** (6 runs, **machine rebooted 11:37** mid-run, resumed the 1 missing run); double-story 'a' glyph experiment (helped d/b/c, hurt a -> **reverted**); terrain **reference panels** discussion; commit-cleanup -> **PR #4 merged** (`47d8586`, master HEAD). |
| **06-05** | Per-particle **phase-space gallery** (CLI + runner opt-in); **2nd main experiment** (9 runs, 3-seed, reproducibility); **ANN control** (TF Sequential, then 5-seed). Committed on branch `claude/experiment-results-and-viz` (`97bd203`, `3738b67`) — **NOT yet in master** (PR #4 was already merged; a new PR is needed). |

---

## 8. Experiment list

| # | experiment | scope | where |
|---|---|---|---|
| E1 | 2x2 ablation | OX/ABC/abcd x {default,improved} x {sigma frozen,learned}, reduced budget (12 runs) | EXPERIMENT_RESULTS.md §1 |
| E2 | CPU/GPU benchmark | ms/epoch per device per dataset (WSL-CPU vs WSL-GPU vs Win-CPU) | §2-3 |
| E3 | 1st main experiment | improved+learned sigma, full budget, seeds 42/1 (6 runs) | §4; research/main_exp_1/ |
| E4 | 'a'-glyph probe | double-story 'a' on abcd, 2 seeds (reverted after) | §5 |
| E5 | 2nd main experiment | improved+learned sigma, full budget, seeds 42/1/2 (9 runs); OX with phase galleries | §6; research/main_exp_2/ |
| E6 | ANN control | TF Sequential MLP, OX/ABC/abcd x 5 seeds (15 runs) | §7; research/ann_control.json |
| (deferred) | FLOPs / compute comparison | CHM (JAX cost_analysis) vs ANN | — |

---

## 9. Guardrails (do not break)

Physics core is sacred: do NOT modify `contact_rhs`, the RBF `V`/grad, the loss
structure (position + lambda_p*momentum), the phase-volume R^2 formula, or the
checkpointed BPTT. Keep the landscape a sum of readable Gaussians (white-box).
Attractor `mu`/`w` stay frozen. All hyperparameters live in `Config`. Don't
commit large binary experiment outputs (gitignore `research/`). Report git
failures verbatim; never force-push.
