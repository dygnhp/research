# Contact Hamiltonian Machine — Algorithm

> Data is turned into physical particles, set in motion on a learned
> potential landscape, and classified by which attractor each ensemble
> converges to. The landscape's complexity grows on demand.

This document explains how the code in `kanzen/` realizes that statement,
mapping every section of the specification onto a concrete module.

---

## 0. Why this approach

A standard neural network produces correct outputs through matrix
multiplications whose intermediate steps carry no physical meaning.  The
*decision* is opaque even when the *answer* is right.

The Contact Hamiltonian Machine (CHM) replaces those multiplications with
ordinary differential equations.  The classifier's entire computation —
preprocessing, forward pass, decision — is a single trajectory through a
2D+1-dimensional phase space governed by a hand-written potential.  The
parameters and the landscape they define remain physical objects:
attractors and barriers we can plot and read.

The trade-off: differentiating an ODE is more delicate than backproping a
ReLU MLP, and we need a convergence guarantee strong enough that
classification by "which attractor" is well-defined.  Contact geometry
provides exactly such a guarantee, as Section 4 explains.

---

## 1. Pipeline overview

```
image (8x8)
    |
    | preprocess.make_S0           # tau filter + N-D lifting + padding
    v
S0  =  [ q(0) | p(0) | z(0) ]      # shape (n_max, 2D+1), with mask
    |
    | dynamics.simulate_diff       # RK4 + lax.scan over n_steps
    v
trajectory  (T+1, n_max, 2D+1)
    |
    | (mask-weighted) CoM at time T
    v
predicted class      argmin_c || CoM(T) - q*_c ||

                                                +- loss.loss_two_class
forward pass + L = ||CoM-q*||^2 + lambda_p |p|^2 -+
                                                +- jax.value_and_grad
                                                +- optax.adamw
        -> growth.PlateauDetector          (detect stagnation)
        -> growth.grow_K  /  growth.grow_D (extend the landscape)
```

Each arrow in the diagram corresponds to exactly one module in `kanzen/`.

---

## 2. Image → particle ensemble  (`preprocess.py`)

A pixel with intensity above `tau` (default `0.5`) becomes a particle.
Background pixels are silently dropped:

```
P = { (r, c) : I(r, c) > tau }
N = |P|
```

Each particle is lifted into `R^D`:

| dim | name              | formula                                                  |
|-----|-------------------|----------------------------------------------------------|
| 0   | `x`               | column index                                             |
| 1   | `y`               | `rows - 1 - row`  (so y grows upward)                    |
| 2   | `z_connectivity`  | `sigmoid(beta * (d_axis - |d_diag_signed|))`             |
| 3   | `local_density`   | 3x3 neighborhood pixel count / 9                         |
| 4+  | placeholder       | 0 (reserved for learned features)                        |

The z-channel separates O-like local structure (axis-connected neighbors)
from X-like local structure (diagonally-connected neighbors).  Without
this third dimension, both classes occupy the same `(x, y)` rectangle and
no learnable potential can route them to distinct attractors.

Initial momentum and contact variable are zero: particles start at rest
and are accelerated only by the landscape's gradient.

**Padding to `n_max`.**  JAX recompiles every time a tensor's shape
changes.  We always allocate the maximum number of slots (`n_max = 64`),
real particles fill the first `N` slots, and a boolean mask marks the
real ones so the dynamics and the loss can ignore the dummy ones.

---

## 3. The RBF terrain  (`terrain.py`, `params.py`)

The potential is a sum of K isotropic Gaussians:

```
V(q) = sum_{k=1..K}  w_k * exp( -||q - mu_k||^2 / (2 * sigma_k^2) )
```

- `w_k < 0` digs an attractor (a well) at `mu_k`.
- `w_k > 0` raises a barrier (a hill) at `mu_k`.
- `sigma_k` is the effective radius of the k-th feature.

The gradient is computed in closed form (no autograd dependence inside
the dynamics, which keeps the RHS explicit and inspectable):

```
grad_q V(q) = - sum_k  w_k * exp(...) * (q - mu_k) / sigma_k^2
```

**Parameter roles**.  The K Gaussians are partitioned into three blocks:

| k                                   | role                | learnable? |
|-------------------------------------|---------------------|------------|
| 0, 1                                | frozen attractors   | no         |
| 2, 3                                | stepping stones     | yes        |
| 4 .. K-1                            | free basis          | yes        |

The frozen attractors are pinned to the class targets `q*_O = (8, 8, ...)`
and `q*_X = (-8, -8, ...)`.  They guarantee that the optimizer is solving
the right classification problem — the attractor coordinates *are* the
class identities.

Stepping stones are placed inside or near the data domain (`(6, 6)` for O
and `(0, 0)` for X) so that on the first epoch every particle already
experiences a non-trivial gradient: a Gaussian at distance > 3 sigma is
numerically dead, and the frozen attractors at distance ~12 from the data
domain would otherwise produce zero force.

`sigma` is stored as `sigma_raw`, with `sigma = softplus(sigma_raw) + 0.1`,
so it is always positive and the optimizer can update it without
constraints.

---

## 4. Contact Hamiltonian dynamics  (`dynamics.py`)

A standard Hamiltonian system on `R^{2N}` conserves phase-space volume
(Liouville's theorem).  Volume preservation is fatal for a classifier:
particles can orbit an attractor forever but they cannot *converge* to
it.

Contact geometry extends the phase space by one variable, `z` (a
dissipation counter), and the *contact Hamiltonian*

```
H_c(q, p, z) = ||p||^2 / 2 + V(q) + gamma * z
```

generates the equations of motion

```
dq/dt = p
dp/dt = - grad V(q) - gamma * p          <-- new damping term
dz/dt = ||p||^2 - H,    H = ||p||^2 / 2 + V(q)
```

The damping `-gamma * p` is not an ad-hoc addition.  It is forced by the
contact bracket structure: `dH_c / dz = gamma`, and the resulting RHS for
`p` automatically includes `-gamma * p`.

**Liouville is intentionally broken.**  The divergence of the new vector
field is

```
div X = - D * gamma   (for D-dimensional q-momentum pairs)
```

so phase-space volume contracts as `exp(-D * gamma * t)`.  This volume
contraction is *the* physical reason particles converge.

**Lyapunov argument for convergence.**  Differentiate the mechanical
energy along trajectories:

```
dH/dt = grad V . dq/dt + p . dp/dt
      = grad V . p     + p . (-grad V - gamma p)
      = - gamma ||p||^2   <=   0
```

So `H` is monotonically non-increasing; equality holds only when `p = 0`,
which by LaSalle's invariance principle implies every trajectory tends to
the set `{p = 0, grad V = 0}` — the stationary points of `V`, i.e. the
attractors.

(The original spec invoked Bendixson's criterion here, which is strictly
a planar result.  LaSalle generalizes cleanly to `R^{2D+1}` and gives
exactly what we need.)

**Implementation.**  The RHS is hand-written in `contact_rhs`.  RK4
(`make_rk4_step`) is jitted and wrapped in `jax.lax.scan` to roll the
trajectory.  Two versions are exported:

- `make_simulate_diff` wraps each scan step with `jax.checkpoint` so
  backpropagation through the full trajectory uses `O(sqrt(T))` memory
  rather than `O(T)`.  This is what training calls.
- `make_simulate_eval` skips the checkpoint and is used by evaluation.

---

## 5. Convergence diagnostics  (`invariants.py`)

Three numbers, computed from a forward trajectory, decide whether the
landscape is good enough:

| symbol     | code                              | meaning                                          |
|------------|-----------------------------------|--------------------------------------------------|
| eps_q      | `epsilon_q`                       | distance from CoM(T) to the class target q*       |
| eps_p      | `epsilon_p`                       | mean |p_i(T)| over real particles                |
| R^2_phase  | `phase_volume_R2`                 | how well log det Cov_{q,p}(t) tracks -D*gamma*t  |

The settle gate (`train._settled`) fires only when *all three* satisfy
their thresholds for *both* classes.  Position-only convergence is
insufficient: a network can be trained to make CoM pass through q* at
exactly time T with residual momentum, in which case the particles
overshoot on the next epoch.  The triple gate prevents that failure mode.

The R^2 score is the spec's "phase-volume contraction R^2 >= 0.90"
condition.  Empirically R^2 is far from 1 during early training (the
landscape pumps energy into the cloud); it climbs as the system settles.

---

## 6. Loss + backpropagation  (`loss.py`)

```
L(theta) = ||CoM_O(T) - q*_O||^2
         + ||CoM_X(T) - q*_X||^2
         + lambda_p * ( mean |p_O(T)|^2 + mean |p_X(T)|^2 )
```

Both classes are evaluated in the same epoch.  The two CoM terms compete
— making the landscape better for one class makes it worse for the other
unless the two attractor basins are genuinely separable.

The gradient is obtained by `jax.value_and_grad` through the full
`lax.scan` trajectory.  Memory cost is controlled by `jax.checkpoint`
inside each scan step (see Section 4); we therefore use plain BPTT
rather than the adjoint method.  Both yield the same gradient up to
discretization; checkpointed BPTT is numerically more robust.

Adam-with-weight-decay over a warmup-cosine LR (`peak=5e-3`, end=1e-5`)
and gradient clipping at global norm 1.0 are the standard recipe.

---

## 7. Autonomous K growth  (`growth.py`)

`PlateauDetector` keeps a sliding window of recent losses.  Plateau =
the relative improvement between the previous window and the current
window is below `plateau_threshold` (default 1%).

When a plateau is detected and the cooldown has elapsed, `grow_K` adds
`K_grow` new learnable RBFs.  Placement is *guided*: for each class that
still has `eps_q > eps_q_thresh`, we drop a new attracting RBF at the
midpoint between the class's mid-trajectory CoM and its target `q*`.

Sign rule.  When a class fails to reach its attractor, the failure mode
is almost always "stuck along the way".  The cure is a new well between
the cloud and the target — hence the negative weight (attractor).  If
both classes are succeeding, we add *paired* RBFs (alternating signs) in
random locations to give the optimizer more degrees of freedom.

After `grow_K` the optimizer state is rebuilt (the param tensors have a
new leading dimension, so Adam's moments must be reinitialized).

---

## 8. Autonomous D growth  (`growth.py`)

When K-growth has been tried `K_grows_before_D` times in a row with no
sustained loss improvement, we go up one dimension.

Three things change at `D -> D+1`:

1. **Particle states gain a new coordinate.**  The lift in
   `preprocess.py` already returns `D`-dimensional q for any D it is
   asked for.  Newly added dimensions for existing centers are 0.
2. **All `mu_k` are zero-padded** in the new dimension.  This preserves
   the existing landscape: at the new dim's zero plane, V is unchanged.
3. **All `sigma_k` are multiplied by `sqrt(D_new / D_old)`.**  In
   `R^D` random pairwise distances scale like `sqrt(D)`; if sigma did
   not scale, every Gaussian would become a delta function as we raise
   D and the gradient would vanish.  This is the Johnson-Lindenstrauss
   justification for the rescaling, and it is exact: after the rescale,
   `||(q - mu)||^2 / (2 sigma^2)` has the same distribution as before
   for points drawn isotropically.

`grow_D` performs (2) and (3); `grow_frozen_D` does the same for the
frozen attractor block.  The resulting full state is rebuilt by
`_build_state(cfg, D_new, prev_params=...)`, which also re-jits the
training step at the new shape.

---

## 9. Classification (Step 3 of the spec)  (`evaluate.py`)

After training settles, `classify(image)` runs a forward simulation and
returns

```
pred = argmin_c  || CoM(T) - q*_c ||,  c in {O, X}
```

`evaluate.py` also implements the standard robustness suite from the
original Block III work:

- `noise_sweep`: flip random pixels and measure accuracy
- `shift_sweep`: translate the image and measure accuracy
- `gamma_sweep`: re-simulate with different damping coefficients
- `ablation_study`: zero out stones / free / both to see which RBFs
  carry the classifier

These are all forward-only.  No training happens during evaluation.

---

## 10. Why CHM is white-box

Three layers of transparency are visible from a finished model:

1. **The computation is a trajectory.**  Every input has an exact
   `(q(t), p(t), z(t))` curve in phase space.  "Why was this pixel
   pattern called X?" has a literal answer: because its center of mass
   reached the basin of attraction at `(-8, -8)`.

2. **The landscape is interpretable.**  Each Gaussian has one weight
   `w_k`.  Sign decides "well or hill"; `sigma_k` decides "wide or
   narrow".  One contour plot summarizes the whole classifier.  Compare
   this to a 4-layer MLP, whose weight matrices encode nothing the eye
   can directly parse.

3. **Convergence is proved, not measured.**  `dH/dt = -gamma ||p||^2`
   holds for every parameter setting.  No matter how much training has
   happened, the system *will* converge to a stationary point of V.
   The only thing training changes is *which* stationary point each
   class lands on.

ANNs say "it works".  CHM says "and here is the equation that says why".

---

## 11. File map

```
kanzen/
  __init__.py        public API
  config.py          Config dataclass: every hyperparameter in one place
  data.py            O/X canonical images + parametric variant generator
  preprocess.py      tau filter + N-D lifting + padded packing into S0
  terrain.py         RBF potential V(q) and analytic grad V(q)
  params.py          frozen attractor block + initial learnable block
  dynamics.py        contact_rhs, RK4, simulate_diff (BPTT) / simulate_eval
  loss.py            phase-space classification loss
  invariants.py      eps_q, eps_p, phase-volume R^2
  growth.py          PlateauDetector, grow_K (guided), grow_D (sqrt(D) sigma)
  train.py           main training loop with autonomous growth
  evaluate.py        classify + noise/shift/gamma/ablation sweeps
  viz.py             6-panel matplotlib summary figure
  main.py            CLI: python -m kanzen.main {train,evaluate,demo}
  ALGORITHM.md       this document
```

Each module is intentionally short and single-purpose; the longest is
`train.py`, because it coordinates everything else.

---

## 12. Running it

```bash
# Smoke test (~100 epochs, no growth)
python -m kanzen.main demo

# Full training
python -m kanzen.main train

# Evaluate the latest run
python -m kanzen.main evaluate
```

Outputs land under `kanzen_runs/run_<timestamp>/`:

```
kanzen_runs/run_20260512_103045/
  params.npz          final w, mu, sigma_raw, D, K_learn
  history.json        loss curve, per-log-interval diagnostics, events
  growth_log.json     epochs and kinds of grow_K / grow_D events
  config.json         the exact Config dataclass used
  summary.png         terrain contours, trajectories, loss, diagnostics
```

---

## 13. What was changed relative to the source spec

| Item                                 | Spec                              | Implementation               | Why                                            |
|--------------------------------------|-----------------------------------|------------------------------|------------------------------------------------|
| Convergence argument for limit-cycle absence | Bendixson's criterion (2D)        | LaSalle invariance (any D)   | Bendixson is strictly planar                  |
| Backprop through the trajectory      | "adjoint method, O(1) memory"     | jax.checkpoint BPTT, O(sqrt(T)) | More robust gradients; only a small memory cost |
| sigma scaling under D growth         | sigma * sqrt(D)                   | sigma * sqrt(D_new/D_old)    | The spec rule is the same rule; we apply it incrementally |
| K-growth sign rule                   | "depends on attractor or barrier" | "attractor when class fails, otherwise alternating pairs" | Spec was schematic; this is one concrete realization |
| Convergence gate                     | eps_q + eps_p                     | eps_q + eps_p + R^2_phase >= 0.90 | The R^2 gate was in Block I but not in the loop; we promote it |

Every other element follows the spec word for word.
