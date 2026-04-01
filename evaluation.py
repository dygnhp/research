"""
=============================================================================
Shared Evaluation Module  --  evaluation.py
=============================================================================
Reusable evaluation functions for any block (II, III, IV, ...).
Each block imports this module instead of duplicating evaluation logic.

Provides:
  - evaluate_single()   : classify one image with given params
  - evaluate_suite()    : classify a dict of images
  - noise_sweep()       : noise robustness
  - shift_sweep()       : translation robustness
  - ablation_study()    : systematically zero RBF groups
  - gamma_sweep()       : damping sensitivity
  - make_eval_figure()  : 8-panel paper-ready figure
  - run_standard_eval() : full evaluation pipeline (all of the above)
=============================================================================
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ═══════════════════════════════════════════════════════════════════════════
# Core evaluation helpers (block-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

def com_single(q_t, mask):
    """Centre-of-mass of real particles at one timestep. Returns (D,)."""
    mf = mask.astype(jnp.float32)
    return jnp.sum(q_t * mf[:, None], axis=0) / jnp.sum(mf)


def classify_traj(trajectory, mask, qO, qX):
    """Classify by final CoM proximity to attractors."""
    qf = com_single(trajectory[-1, :, :qO.shape[0]], mask)
    dO = float(jnp.linalg.norm(qf - qO))
    dX = float(jnp.linalg.norm(qf - qX))
    return ('O' if dO < dX else 'X'), dO, dX, qf


def add_noise(image, n_flips, rng_key):
    """Flip n_flips random pixels."""
    flat = image.flatten().copy()
    indices = np.random.RandomState(int(rng_key)).choice(
        len(flat), size=min(n_flips, len(flat)), replace=False)
    flat[indices] = 1.0 - flat[indices]
    return flat.reshape(image.shape)


def shift_image(image, dx, dy):
    """Shift image by (dx, dy) pixels, zero-padding."""
    result = np.zeros_like(image)
    rows, cols = image.shape
    for r in range(rows):
        for c in range(cols):
            sr, sc = r - dy, c - dx
            if 0 <= sr < rows and 0 <= sc < cols:
                result[r, c] = image[sr, sc]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation functions (parameterized by simulator + preprocessor)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_single(image, w, mu, sigma, preprocess_fn, simulate_fn,
                    qO, qX, label="?", conv_q_thr=2.0, conv_p_thr=0.5,
                    gamma=None, sim_kwargs=None):
    """
    Evaluate a single image. Block-agnostic: accepts preprocess_fn and
    simulate_fn as callables.

    Args:
        image         : (H, W) binary image
        w, mu, sigma  : RBF parameters (full, including frozen)
        preprocess_fn : image -> (S0, mask)  [block-specific]
        simulate_fn   : (S0, w, mu, sigma, **kw) -> trajectory
        qO, qX        : attractor targets
        label         : string label
        gamma         : override damping (passed via sim_kwargs)
        sim_kwargs    : extra kwargs for simulate_fn (e.g. gamma=...)
    """
    S0, mask = preprocess_fn(image)
    n_particles = int(mask.sum())
    if n_particles == 0:
        return {'label': label, 'pred': '?', 'dist_O': float('inf'),
                'dist_X': float('inf'), 'final_com': [0]*qO.shape[0],
                'n_particles': 0, 'eps_q': float('inf'), 'eps_p': float('inf'),
                'converged_q': False, 'converged_p': False}

    kw = dict(sim_kwargs or {})
    if gamma is not None:
        kw['gamma'] = gamma
    traj = simulate_fn(S0, w, mu, sigma, **kw)
    D = qO.shape[0]
    pred, dO, dX, qf = classify_traj(traj, mask, qO, qX)

    mf = mask.astype(jnp.float32)
    p_final = traj[-1, :, D:2*D]
    eps_p = float(jnp.sum(jnp.sum(p_final ** 2, axis=-1) * mf) / jnp.sum(mf))
    eps_q_O = float(jnp.linalg.norm(qf - qO))
    eps_q_X = float(jnp.linalg.norm(qf - qX))
    eps_q = min(eps_q_O, eps_q_X)

    return {
        'label':       label,
        'pred':        pred,
        'dist_O':      dO,
        'dist_X':      dX,
        'final_com':   [float(qf[i]) for i in range(D)],
        'n_particles': n_particles,
        'eps_q':       eps_q,
        'eps_p':       eps_p,
        'converged_q': eps_q < conv_q_thr,
        'converged_p': eps_p < conv_p_thr,
    }


def evaluate_suite(images_dict, w, mu, sigma, preprocess_fn, simulate_fn,
                   qO, qX, gamma=None, sim_kwargs=None):
    """Evaluate a dict of {name: image}."""
    results = []
    for name, img in images_dict.items():
        r = evaluate_single(img, w, mu, sigma, preprocess_fn, simulate_fn,
                            qO, qX, label=name, gamma=gamma, sim_kwargs=sim_kwargs)
        results.append(r)
        com_str = ",".join(f"{c:.2f}" for c in r['final_com'][:2])
        print(f"    {name:>10s}: pred={r['pred']}  "
              f"dO={r['dist_O']:.3f}  dX={r['dist_X']:.3f}  "
              f"com=({com_str})  n={r['n_particles']}")
    return results


def noise_sweep(base_image, true_label, w, mu, sigma,
                preprocess_fn, simulate_fn, qO, qX,
                flip_counts=(0, 1, 2, 3, 4, 5, 6, 8, 10), n_trials=5):
    """Noise robustness sweep."""
    print(f"\n  Noise sweep on {true_label}-image ({n_trials} trials/level):")
    sweep = []
    for n_flips in flip_counts:
        correct = 0
        for trial in range(n_trials):
            img = base_image if n_flips == 0 else add_noise(
                base_image, n_flips, rng_key=trial * 100 + n_flips)
            r = evaluate_single(img, w, mu, sigma, preprocess_fn, simulate_fn,
                                qO, qX, label=f"{true_label}_n{n_flips}")
            if r['pred'] == true_label:
                correct += 1
        acc = correct / n_trials
        sweep.append((n_flips, acc))
        print(f"    flips={n_flips:2d}: acc={acc:.0%} ({correct}/{n_trials})")
    return sweep


def shift_sweep(base_image, true_label, w, mu, sigma,
                preprocess_fn, simulate_fn, qO, qX,
                shifts=(-2, -1, 0, 1, 2)):
    """Translation robustness sweep."""
    print(f"\n  Shift sweep on {true_label}-image:")
    results = {}
    for dx in shifts:
        for dy in shifts:
            img = shift_image(base_image, dx, dy)
            if int(img.sum()) == 0:
                continue
            r = evaluate_single(img, w, mu, sigma, preprocess_fn, simulate_fn,
                                qO, qX, label=f"{true_label}_s({dx},{dy})")
            results[(dx, dy)] = r
            status = 'OK' if r['pred'] == true_label else 'MISS'
            print(f"    shift=({dx:+d},{dy:+d}): pred={r['pred']} "
                  f"dO={r['dist_O']:.2f} dX={r['dist_X']:.2f}  [{status}]")
    total = len(results)
    correct = sum(1 for r in results.values() if r['pred'] == true_label)
    print(f"    Shift accuracy: {correct}/{total} = {correct/max(total,1):.0%}")
    return results


def ablation_study(params, full_params_fn, images_dict,
                   preprocess_fn, simulate_fn, qO, qX, K=16):
    """
    Ablation: full / no stepping stones / no free / attractors only.
    full_params_fn: params_dict -> (w, mu, sigma)
    """
    print("\n" + "-" * 50)
    print("ABLATION STUDY")
    print("-" * 50)
    ablations = {}

    w, mu, sigma = full_params_fn(params)

    # (a) Full model
    print("\n  [A] Full model (all RBFs):")
    ablations['full'] = evaluate_suite(
        images_dict, w, mu, sigma, preprocess_fn, simulate_fn, qO, qX)

    # (b) No stepping stones (zero k=2,3)
    print("\n  [B] No stepping stones (w_2=w_3=0):")
    w_ns = w.at[2].set(0.0).at[3].set(0.0)
    ablations['no_stones'] = evaluate_suite(
        images_dict, w_ns, mu, sigma, preprocess_fn, simulate_fn, qO, qX)

    # (c) No free RBFs (zero k=4..K-1)
    print(f"\n  [C] No free RBFs (w_4..w_{K-1}=0):")
    w_nf = w
    for k in range(4, K):
        w_nf = w_nf.at[k].set(0.0)
    ablations['no_free'] = evaluate_suite(
        images_dict, w_nf, mu, sigma, preprocess_fn, simulate_fn, qO, qX)

    # (d) Attractors only
    print("\n  [D] Attractors only (w_0, w_1 only):")
    w_ao = jnp.zeros_like(w).at[0].set(w[0]).at[1].set(w[1])
    ablations['attract_only'] = evaluate_suite(
        images_dict, w_ao, mu, sigma, preprocess_fn, simulate_fn, qO, qX)

    return ablations


def gamma_sweep(params, full_params_fn, preprocess_fn, simulate_fn,
                O_image, X_image, qO, qX,
                gamma_values=(0.5, 1.0, 1.5, 2.0, 3.0)):
    """Evaluate classification with different damping coefficients."""
    print("\n" + "-" * 50)
    print("GAMMA SENSITIVITY ANALYSIS")
    print("-" * 50)
    w, mu, sigma = full_params_fn(params)
    results = {}
    for g in gamma_values:
        r_O = evaluate_single(O_image, w, mu, sigma, preprocess_fn, simulate_fn,
                              qO, qX, label='O', gamma=g)
        r_X = evaluate_single(X_image, w, mu, sigma, preprocess_fn, simulate_fn,
                              qO, qX, label='X', gamma=g)
        ok_O = (r_O['pred'] == 'O')
        ok_X = (r_X['pred'] == 'X')
        results[g] = {'O': r_O, 'X': r_X, 'both_correct': ok_O and ok_X}
        print(f"  gamma={g:.1f}: O->{r_O['pred']}({'OK' if ok_O else 'FAIL'}) "
              f"X->{r_X['pred']}({'OK' if ok_X else 'FAIL'})  "
              f"dO={r_O['dist_O']:.3f}  dX={r_X['dist_X']:.3f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Figure generation
# ═══════════════════════════════════════════════════════════════════════════

def make_eval_figure(params, full_params_fn, rbf_potential_fn,
                     simulate_fn, preprocess_fn,
                     O_image, X_image, qO, qX, mask_O, mask_X,
                     noise_O, noise_X, shift_O, shift_X,
                     ablation_results, gamma_results, novel_results,
                     title_prefix="Evaluation", output_path="eval_figure.png",
                     gamma_default=1.5, T_final=10.0, dt=0.05, K=16, N_steps=200):
    """
    8-panel evaluation figure, parameterized for any block.
    """
    w, mu, sigma = full_params_fn(params)
    mu_np = np.array(mu)
    w_np  = np.array(w)
    mO = np.array(mask_O, dtype=bool)
    mX = np.array(mask_X, dtype=bool)

    fig = plt.figure(figsize=(22, 10))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.40)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

    # [1,1] Potential landscape (project to first 2 dims, slice at midpoint)
    ax = axes[0][0]
    D = mu.shape[1]
    xy_r = np.linspace(-12, 12, 200)
    gx, gy = np.meshgrid(xy_r, xy_r)
    fill_vals = [0.5] * max(D - 2, 0)
    cols = [gx.ravel(), gy.ravel()] + [np.full(gx.size, v) for v in fill_vals]
    q_grid = jnp.array(np.stack(cols, axis=1).astype(np.float32))
    Vg = np.array(rbf_potential_fn(q_grid, w, mu, sigma)).reshape(gx.shape)
    cf = ax.contourf(gx, gy, Vg, levels=40, cmap='RdYlBu_r', alpha=0.85)
    ax.contour(gx, gy, Vg, levels=20, colors='k', linewidths=0.3, alpha=0.4)
    plt.colorbar(cf, ax=ax, shrink=0.85)
    ax.scatter(*np.array(qO)[:2], s=200, c='blue', marker='*', zorder=6)
    ax.scatter(*np.array(qX)[:2], s=200, c='red',  marker='*', zorder=6)
    for k in range(min(K, mu_np.shape[0])):
        col = 'white' if k < 2 else ('cyan' if k < 4 else 'lime')
        ax.scatter(*mu_np[k, :2], s=15, c=col, edgecolors='k',
                   linewidths=0.3, zorder=5)
    ax.set(xlim=(-12, 12), ylim=(-12, 12), xlabel='x', ylabel='y',
           title=f'RBF Potential (D={D}, K={K})')

    # [1,2] Noise robustness
    ax = axes[0][1]
    ax.plot([s[0] for s in noise_O], [s[1] for s in noise_O],
            'bo-', lw=2, label='O-image')
    ax.plot([s[0] for s in noise_X], [s[1] for s in noise_X],
            'rs-', lw=2, label='X-image')
    ax.set(xlabel='# Pixel Flips', ylabel='Accuracy',
           title='Noise Robustness', ylim=(-0.05, 1.15))
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.legend(fontsize=9)

    # [1,3] Shift robustness
    ax = axes[0][2]
    shifts = sorted(set(k[0] for k in shift_O.keys()))
    n_s = len(shifts)
    heat = np.zeros((n_s, n_s))
    for i, dy in enumerate(shifts):
        for j, dx in enumerate(shifts):
            ok_O = shift_O.get((dx, dy), {}).get('pred', '?') == 'O'
            ok_X = shift_X.get((dx, dy), {}).get('pred', '?') == 'X'
            heat[i, j] = int(ok_O) + int(ok_X)
    im = ax.imshow(heat, cmap='RdYlGn', vmin=0, vmax=2,
                   extent=[shifts[0]-0.5, shifts[-1]+0.5,
                           shifts[0]-0.5, shifts[-1]+0.5], origin='lower')
    plt.colorbar(im, ax=ax, ticks=[0, 1, 2], label='Correct (0-2)')
    ax.set(xlabel='dx shift', ylabel='dy shift', title='Shift Robustness')

    # [1,4] Novel patterns
    ax = axes[0][3]
    ax.axis('off')
    lines = ["Novel Pattern Classification", "=" * 30, ""]
    for r in novel_results:
        lines.append(f"  {r['label']:>6s} -> {r['pred']}  "
                     f"dO={r['dist_O']:.2f}  dX={r['dist_X']:.2f}")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=8.5, fontfamily='monospace', va='top')
    ax.set_title('Novel Patterns', fontweight='bold')

    # [2,1] Ablation bar chart
    ax = axes[1][0]
    configs = ['full', 'no_stones', 'no_free', 'attract_only']
    labels  = ['Full\nmodel', 'No step\nstones', 'No free\nRBFs', 'Attractors\nonly']
    accuracies = []
    for cfg in configs:
        res_list = ablation_results.get(cfg, [])
        n_correct = sum(1 for r in res_list
                        if (r['label'] == 'O' and r['pred'] == 'O') or
                           (r['label'] == 'X' and r['pred'] == 'X'))
        n_total = sum(1 for r in res_list if r['label'] in ('O', 'X'))
        accuracies.append(n_correct / max(n_total, 1))
    colors = ['#2196F3', '#FF9800', '#F44336', '#9E9E9E']
    ax.bar(range(len(configs)), accuracies, color=colors, edgecolor='k', lw=0.5)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set(ylabel='Accuracy', ylim=(0, 1.15), title='Ablation Study')
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=9, fontweight='bold')

    # [2,2] Gamma sensitivity
    ax = axes[1][1]
    gammas = sorted(gamma_results.keys())
    dO_vals = [gamma_results[g]['O']['dist_O'] for g in gammas]
    dX_vals = [gamma_results[g]['X']['dist_X'] for g in gammas]
    ax.plot(gammas, dO_vals, 'bo-', lw=2, label='O: dist to q*_O')
    ax.plot(gammas, dX_vals, 'rs-', lw=2, label='X: dist to q*_X')
    ax.set(xlabel='gamma', ylabel='Distance to target', title='Gamma Sensitivity')
    ax.legend(fontsize=8)

    # [2,3] Final trajectories
    ax = axes[1][2]
    S0_O, _ = preprocess_fn(O_image)
    S0_X, _ = preprocess_fn(X_image)
    traj_O = simulate_fn(S0_O, w, mu, sigma)
    traj_X = simulate_fn(S0_X, w, mu, sigma)
    tOnp = np.array(traj_O[:, mO, :])
    tXnp = np.array(traj_X[:, mX, :])
    dec = max(1, N_steps // 50)
    for i in range(tOnp.shape[1]):
        ax.plot(tOnp[::dec, i, 0], tOnp[::dec, i, 1], 'b-', alpha=0.15, lw=0.5)
    for i in range(tXnp.shape[1]):
        ax.plot(tXnp[::dec, i, 0], tXnp[::dec, i, 1], 'r-', alpha=0.15, lw=0.5)
    ax.scatter(*np.array(qO)[:2], s=200, c='blue', marker='*', zorder=6)
    ax.scatter(*np.array(qX)[:2], s=200, c='red',  marker='*', zorder=6)
    ax.set(xlabel='q_x', ylabel='q_y', title='Particle Trajectories')
    ax.legend(['O particles', 'X particles'], fontsize=8)

    # [2,4] Summary
    ax = axes[1][3]
    ax.axis('off')
    full_res = ablation_results.get('full', [])
    o_ok = any(r['label'] == 'O' and r['pred'] == 'O' for r in full_res)
    x_ok = any(r['label'] == 'X' and r['pred'] == 'X' for r in full_res)
    cls_pass = o_ok and x_ok
    noise_robust = (all(s[1] >= 0.8 for s in noise_O[:4]) and
                    all(s[1] >= 0.8 for s in noise_X[:4]))
    n_gamma_ok = sum(1 for g in gamma_results.values() if g['both_correct'])
    all_pass = cls_pass and noise_robust
    bg = '#d4edda' if all_pass else '#fff3cd'
    ax.set_facecolor(bg)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    lns = [
        f"{title_prefix} SUMMARY",
        "=" * 32, "",
        f"Classification: {'PASS' if cls_pass else 'FAIL'}",
        f"Noise robust (<=3 flips): {'PASS' if noise_robust else 'FAIL'}",
        f"Gamma: {n_gamma_ok}/{len(gamma_results)} correct",
        f"Novel patterns: {len(novel_results)}",
        f"D={D}  K={K}",
        "",
        f"OVERALL: {'*** PASS ***' if all_pass else '--- PARTIAL ---'}",
    ]
    ax.text(0.05, 0.95, "\n".join(lns), transform=ax.transAxes,
            fontsize=8, fontfamily='monospace', va='top')
    ax.set_title('Summary', fontweight='bold',
                 color='darkgreen' if all_pass else 'darkorange')

    fig.suptitle(f"{title_prefix}  |  D={D}  K={K}  gamma={gamma_default}  "
                 f"T={T_final}  dt={dt}", fontsize=10.5, y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  [Figure] Saved -> {output_path}")
    plt.close(fig)
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════
# Standard evaluation pipeline (convenience wrapper)
# ═══════════════════════════════════════════════════════════════════════════

def run_standard_eval(params, full_params_fn, rbf_potential_fn,
                      preprocess_fn, simulate_fn,
                      O_image, X_image, qO, qX, mask_O, mask_X,
                      novel_images=None, dataset_fn=None,
                      title_prefix="Evaluation", output_path="eval_figure.png",
                      gamma_default=1.5, T_final=10.0, dt=0.05, K=16, N_steps=200):
    """
    Full evaluation pipeline. Returns (params, all_pass, results_dict).
    """
    w, mu, sigma = full_params_fn(params)
    print(f"\n  K={K} D={mu.shape[1]} params ready.  "
          f"sigma=[{float(jnp.min(sigma)):.3f}, {float(jnp.max(sigma)):.3f}]")

    # 1. Baseline
    print("\n[Eval 1] Baseline (canonical O/X):")
    baseline = {'O': O_image, 'X': X_image}
    baseline_results = evaluate_suite(
        baseline, w, mu, sigma, preprocess_fn, simulate_fn, qO, qX)

    # 2. Dataset variants (if provided)
    ds_acc = None
    if dataset_fn is not None:
        print("\n[Eval 2] Dataset-variant evaluation:")
        eval_ds = dataset_fn()
        n_ok_O = sum(1 for img in eval_ds['O_images']
                     if evaluate_single(img, w, mu, sigma, preprocess_fn,
                                        simulate_fn, qO, qX)['pred'] == 'O')
        n_ok_X = sum(1 for img in eval_ds['X_images']
                     if evaluate_single(img, w, mu, sigma, preprocess_fn,
                                        simulate_fn, qO, qX)['pred'] == 'X')
        n_ds = eval_ds['n_per_class']
        ds_acc = (n_ok_O + n_ok_X) / (2 * n_ds)
        print(f"    O: {n_ok_O}/{n_ds}  X: {n_ok_X}/{n_ds}  "
              f"Overall: {ds_acc:.0%}")

    # 3-4. Robustness
    print("\n[Eval 3] Noise robustness:")
    noise_O = noise_sweep(O_image, 'O', w, mu, sigma,
                          preprocess_fn, simulate_fn, qO, qX)
    noise_X = noise_sweep(X_image, 'X', w, mu, sigma,
                          preprocess_fn, simulate_fn, qO, qX)

    print("\n[Eval 4] Shift robustness:")
    shift_O = shift_sweep(O_image, 'O', w, mu, sigma,
                          preprocess_fn, simulate_fn, qO, qX)
    shift_X = shift_sweep(X_image, 'X', w, mu, sigma,
                          preprocess_fn, simulate_fn, qO, qX)

    # 5. Novel patterns
    novel_results = []
    if novel_images:
        print("\n[Eval 5] Novel patterns:")
        novel_results = evaluate_suite(
            novel_images, w, mu, sigma, preprocess_fn, simulate_fn, qO, qX)

    # 6-7. Ablation + Gamma
    print("\n[Eval 6] Ablation:")
    ablation_results = ablation_study(
        params, full_params_fn, baseline, preprocess_fn, simulate_fn, qO, qX, K)

    print("\n[Eval 7] Gamma sensitivity:")
    gamma_results = gamma_sweep(
        params, full_params_fn, preprocess_fn, simulate_fn,
        O_image, X_image, qO, qX)

    # 8. Figure
    print("\n[Eval 8] Generating figure...")
    all_pass = make_eval_figure(
        params, full_params_fn, rbf_potential_fn,
        simulate_fn, preprocess_fn,
        O_image, X_image, qO, qX, mask_O, mask_X,
        noise_O, noise_X, shift_O, shift_X,
        ablation_results, gamma_results, novel_results,
        title_prefix=title_prefix, output_path=output_path,
        gamma_default=gamma_default, T_final=T_final, dt=dt, K=K, N_steps=N_steps)

    return params, all_pass, {
        'baseline': baseline_results,
        'ds_accuracy': ds_acc,
        'noise_O': noise_O, 'noise_X': noise_X,
        'shift_O': shift_O, 'shift_X': shift_X,
        'novel': novel_results,
        'ablation': ablation_results,
        'gamma': gamma_results,
    }
