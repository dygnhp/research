"""
=============================================================================
3D Terrain Visualizer -- terrain_viz.py
=============================================================================
Real-time 3D surface plot of the RBF potential landscape that evolves
during training. Shows how the learned potential changes epoch by epoch.

Modes:
  1. Live training: attach to Block II/III training loop, update each epoch
  2. Replay: load saved checkpoints and animate the terrain evolution
  3. Snapshot: single static 3D plot of current potential

Usage:
    # Standalone replay from checkpoints
    python terrain_viz.py

    # From training code (live mode)
    from terrain_viz import TerrainVisualizer
    viz = TerrainVisualizer(mode='live')
    for epoch in training_loop:
        viz.update(w, mu, sigma, epoch, loss)
    viz.save_animation("terrain_evolution.gif")

    # Quick snapshot
    from terrain_viz import snapshot_3d
    snapshot_3d(w, mu, sigma, output_path="terrain.png")
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
import os
import sys

_HERE = Path(__file__).resolve().parent

# Try to import JAX (optional for replay mode with pre-computed frames)
try:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# ===========================================================================
# Core: compute potential on a grid (numpy fallback + JAX)
# ===========================================================================

def rbf_potential_np(q_grid, w, mu, sigma):
    """
    Compute RBF potential on a grid using pure numpy.
    q_grid : (M, D)   grid points
    w      : (K,)     weights
    mu     : (K, D)   centers
    sigma  : (K,)     widths
    Returns: (M,)
    """
    w = np.asarray(w)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    diff = q_grid[:, None, :] - mu[None, :, :]       # (M, K, D)
    sq_dist = np.sum(diff ** 2, axis=-1)               # (M, K)
    gauss = np.exp(-sq_dist / (2.0 * sigma ** 2))     # (M, K)
    return np.sum(w * gauss, axis=-1)                   # (M,)


def make_xy_grid(x_range=(-12, 12), y_range=(-12, 12), resolution=150):
    """Create meshgrid and flat query array."""
    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy


def compute_terrain(w, mu, sigma, gx, gy, z_slice=0.5):
    """Compute potential surface V(x, y, z_slice)."""
    D = np.asarray(mu).shape[1]
    flat_xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    fill = np.full((flat_xy.shape[0], max(D - 2, 0)), z_slice)
    q_grid = np.concatenate([flat_xy, fill], axis=1).astype(np.float32)
    V = rbf_potential_np(q_grid, w, mu, sigma)
    return V.reshape(gx.shape)


# ===========================================================================
# Snapshot: single 3D terrain plot
# ===========================================================================

def snapshot_3d(w, mu, sigma, output_path="terrain_3d.png",
                title="RBF Potential Landscape", z_slice=0.5,
                x_range=(-12, 12), y_range=(-12, 12), resolution=200,
                elevation=35, azimuth=-60, show_rbf_centers=True,
                show_attractors=True):
    """
    Create a single 3D surface plot of the potential landscape.

    Args:
        w, mu, sigma : RBF parameters (numpy or JAX arrays)
        output_path  : where to save the figure
        title        : figure title
        z_slice      : z-value for the slice (default 0.5)
        elevation    : camera elevation angle
        azimuth      : camera azimuth angle
        show_rbf_centers : plot RBF center locations on the surface
        show_attractors  : mark O/X attractor positions
    """
    w_np = np.asarray(w)
    mu_np = np.asarray(mu)
    sigma_np = np.asarray(sigma)

    gx, gy = make_xy_grid(x_range, y_range, resolution)
    V = compute_terrain(w_np, mu_np, sigma_np, gx, gy, z_slice)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Surface
    V_clipped = np.clip(V, np.percentile(V, 1), np.percentile(V, 99))
    surf = ax.plot_surface(gx, gy, V_clipped, cmap='RdYlBu_r',
                           alpha=0.85, linewidth=0, antialiased=True,
                           rstride=2, cstride=2)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, label='V(q)')

    # RBF centers
    if show_rbf_centers:
        K = mu_np.shape[0]
        for k in range(K):
            x_c, y_c = mu_np[k, 0], mu_np[k, 1]
            if x_range[0] <= x_c <= x_range[1] and y_range[0] <= y_c <= y_range[1]:
                # Compute V at center
                q_c = np.zeros((1, mu_np.shape[1]), dtype=np.float32)
                q_c[0] = mu_np[k]
                v_c = rbf_potential_np(q_c, w_np, mu_np, sigma_np)[0]
                color = 'white' if k < 2 else ('cyan' if k < 4 else 'lime')
                size = 80 if k < 2 else (50 if k < 4 else 25)
                ax.scatter([x_c], [y_c], [v_c], c=color, s=size,
                           edgecolors='black', linewidths=0.5, zorder=10,
                           depthshade=False)

    # Attractors
    if show_attractors:
        for pos, color, label in [([8, 8], 'blue', 'q*_O'),
                                   ([-8, -8], 'red', 'q*_X')]:
            q_a = np.zeros((1, mu_np.shape[1]), dtype=np.float32)
            q_a[0, 0], q_a[0, 1] = pos[0], pos[1]
            if mu_np.shape[1] >= 3:
                q_a[0, 2] = z_slice
            v_a = rbf_potential_np(q_a, w_np, mu_np, sigma_np)[0]
            ax.scatter([pos[0]], [pos[1]], [v_a], c=color, s=200,
                       marker='*', zorder=15, depthshade=False, label=label)

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('V(q)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.view_init(elev=elevation, azim=azimuth)

    if show_attractors:
        ax.legend(loc='upper left', fontsize=9)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Terrain] Saved -> {output_path}")
    return output_path


# ===========================================================================
# TerrainVisualizer: captures frames during training for animation
# ===========================================================================

class TerrainVisualizer:
    """
    Captures terrain snapshots during training and produces animations.

    Usage:
        viz = TerrainVisualizer()
        for epoch in training:
            if epoch % viz.capture_every == 0:
                viz.update(w, mu, sigma, epoch, loss)
        viz.save_animation("evolution.gif")
        viz.save_filmstrip("filmstrip.png")
    """

    def __init__(self, x_range=(-12, 12), y_range=(-12, 12),
                 resolution=100, z_slice=0.5, capture_every=1,
                 elevation=35, azimuth=-60):
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.z_slice = z_slice
        self.capture_every = capture_every
        self.elevation = elevation
        self.azimuth = azimuth

        self.gx, self.gy = make_xy_grid(x_range, y_range, resolution)
        self.frames = []  # list of (epoch, loss, V_surface, w, mu, sigma)

    def update(self, w, mu, sigma, epoch, loss=None):
        """Capture a terrain frame."""
        w_np = np.asarray(w).copy()
        mu_np = np.asarray(mu).copy()
        sigma_np = np.asarray(sigma).copy()
        V = compute_terrain(w_np, mu_np, sigma_np, self.gx, self.gy,
                            self.z_slice)
        self.frames.append({
            'epoch': epoch,
            'loss': float(loss) if loss is not None else None,
            'V': V.copy(),
            'w': w_np, 'mu': mu_np, 'sigma': sigma_np,
        })

    def _render_frame(self, frame_data, ax, vmin, vmax):
        """Render a single frame onto an Axes3D."""
        ax.clear()
        V_clip = np.clip(frame_data['V'], vmin, vmax)
        ax.plot_surface(self.gx, self.gy, V_clip, cmap='RdYlBu_r',
                        alpha=0.85, linewidth=0, antialiased=True,
                        rstride=3, cstride=3, vmin=vmin, vmax=vmax)

        mu = frame_data['mu']
        w = frame_data['w']
        K = mu.shape[0]
        for k in range(min(K, 4)):  # show frozen + stones only for clarity
            x_c, y_c = mu[k, 0], mu[k, 1]
            if (self.x_range[0] <= x_c <= self.x_range[1] and
                    self.y_range[0] <= y_c <= self.y_range[1]):
                q_c = np.zeros((1, mu.shape[1]), dtype=np.float32)
                q_c[0] = mu[k]
                v_c = rbf_potential_np(q_c, w, mu, frame_data['sigma'])[0]
                v_c = np.clip(v_c, vmin, vmax)
                color = 'white' if k < 2 else 'cyan'
                ax.scatter([x_c], [y_c], [v_c], c=color, s=60,
                           edgecolors='black', linewidths=0.5, zorder=10,
                           depthshade=False)

        # Attractors
        for pos, color in [([8, 8], 'blue'), ([-8, -8], 'red')]:
            q_a = np.zeros((1, mu.shape[1]), dtype=np.float32)
            q_a[0, 0], q_a[0, 1] = pos[0], pos[1]
            if mu.shape[1] >= 3:
                q_a[0, 2] = self.z_slice
            v_a = rbf_potential_np(q_a, w, mu, frame_data['sigma'])[0]
            v_a = np.clip(v_a, vmin, vmax)
            ax.scatter([pos[0]], [pos[1]], [v_a], c=color, s=150,
                       marker='*', zorder=15, depthshade=False)

        ep = frame_data['epoch']
        loss_s = f"  loss={frame_data['loss']:.2f}" if frame_data['loss'] else ""
        K = len(frame_data['w'])
        D = frame_data['mu'].shape[1]
        ax.set_title(f"Epoch {ep}  D={D}  K={K}{loss_s}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('V')
        ax.view_init(elev=self.elevation, azim=self.azimuth)

    def save_animation(self, output_path="terrain_evolution.gif", fps=4,
                       dpi=100):
        """Save captured frames as an animated GIF."""
        if not self.frames:
            print("  [Terrain] No frames to animate.")
            return None

        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            print("  [Terrain] matplotlib animation not available.")
            return None

        # Global V range across all frames
        all_V = np.concatenate([f['V'].ravel() for f in self.frames])
        vmin = np.percentile(all_V, 2)
        vmax = np.percentile(all_V, 98)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        def animate(i):
            self._render_frame(self.frames[i], ax, vmin, vmax)

        anim = FuncAnimation(fig, animate, frames=len(self.frames),
                             interval=1000 // fps, repeat=True)
        anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        print(f"  [Terrain] Animation saved -> {output_path} "
              f"({len(self.frames)} frames)")
        return output_path

    def save_filmstrip(self, output_path="terrain_filmstrip.png",
                       n_panels=8, dpi=120):
        """Save key frames as a horizontal filmstrip."""
        if not self.frames:
            print("  [Terrain] No frames.")
            return None

        # Select evenly spaced frames
        n = min(n_panels, len(self.frames))
        indices = np.linspace(0, len(self.frames) - 1, n, dtype=int)
        selected = [self.frames[i] for i in indices]

        all_V = np.concatenate([f['V'].ravel() for f in selected])
        vmin = np.percentile(all_V, 2)
        vmax = np.percentile(all_V, 98)

        fig = plt.figure(figsize=(5 * n, 4.5))
        for i, frame in enumerate(selected):
            ax = fig.add_subplot(1, n, i + 1, projection='3d')
            self._render_frame(frame, ax, vmin, vmax)

        fig.suptitle("RBF Potential Landscape Evolution",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [Terrain] Filmstrip saved -> {output_path} ({n} panels)")
        return output_path

    def save_comparison(self, output_path="terrain_comparison.png", dpi=150):
        """Save first vs last frame side-by-side with difference."""
        if len(self.frames) < 2:
            print("  [Terrain] Need >= 2 frames for comparison.")
            return None

        first, last = self.frames[0], self.frames[-1]
        V_diff = last['V'] - first['V']

        all_V = np.concatenate([first['V'].ravel(), last['V'].ravel()])
        vmin = np.percentile(all_V, 2)
        vmax = np.percentile(all_V, 98)

        fig = plt.figure(figsize=(18, 5.5))

        # First frame
        ax1 = fig.add_subplot(131, projection='3d')
        self._render_frame(first, ax1, vmin, vmax)
        ax1.set_title(f"Initial (epoch {first['epoch']})", fontweight='bold')

        # Last frame
        ax2 = fig.add_subplot(132, projection='3d')
        self._render_frame(last, ax2, vmin, vmax)
        ax2.set_title(f"Final (epoch {last['epoch']})", fontweight='bold')

        # Difference (2D heatmap)
        ax3 = fig.add_subplot(133)
        abs_max = max(abs(V_diff.min()), abs(V_diff.max()))
        im = ax3.imshow(V_diff, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max,
                        extent=[*self.x_range, *self.y_range], origin='lower')
        plt.colorbar(im, ax=ax3, label='V_final - V_initial')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('Potential Change (2D)', fontweight='bold')

        # Mark attractors on diff
        ax3.scatter(8, 8, s=150, c='blue', marker='*', zorder=5)
        ax3.scatter(-8, -8, s=150, c='red', marker='*', zorder=5)

        fig.suptitle("Terrain Evolution: Before vs After Training",
                     fontsize=13, fontweight='bold')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  [Terrain] Comparison saved -> {output_path}")
        return output_path


# ===========================================================================
# Replay from checkpoints
# ===========================================================================

def replay_from_checkpoints(checkpoint_dir, output_dir=None,
                            full_params_fn=None):
    """
    Load saved .npy checkpoint files and create terrain animation.

    Args:
        checkpoint_dir : directory containing block*_params_*.npy files
        output_dir     : where to save outputs (default: checkpoint_dir)
        full_params_fn : function to convert learnable -> full params
                         If None, assumes checkpoints contain full arrays.
    """
    ckpt_dir = Path(checkpoint_dir)
    out_dir = Path(output_dir) if output_dir else ckpt_dir

    # Find checkpoint files
    npy_files = sorted(ckpt_dir.glob("*params*.npy"))
    if not npy_files:
        print(f"  [Terrain] No checkpoint files found in {ckpt_dir}")
        return None

    print(f"  [Terrain] Found {len(npy_files)} checkpoints in {ckpt_dir}")
    viz = TerrainVisualizer(resolution=100)

    for f in npy_files:
        data = np.load(str(f), allow_pickle=True).item()

        if 'w_full' in data:
            w, mu, sigma = data['w_full'], data['mu_full'], data['sigma_full']
        elif full_params_fn is not None:
            params = {k: data[k] for k in ('w', 'mu', 'sigma_raw') if k in data}
            w, mu, sigma = full_params_fn(params)
            w, mu, sigma = np.asarray(w), np.asarray(mu), np.asarray(sigma)
        else:
            w, mu, sigma = data.get('w'), data.get('mu'), data.get('sigma')
            if w is None:
                print(f"    Skipping {f.name} (no param arrays)")
                continue

        epoch = data.get('epoch', 0)
        loss = data.get('loss', None)
        viz.update(w, mu, sigma, epoch, loss)
        print(f"    Loaded {f.name}: epoch={epoch} K={len(w)} D={mu.shape[1]}")

    # Generate outputs
    viz.save_filmstrip(str(out_dir / "terrain_filmstrip.png"))
    viz.save_animation(str(out_dir / "terrain_evolution.gif"), fps=2)
    if len(viz.frames) >= 2:
        viz.save_comparison(str(out_dir / "terrain_comparison.png"))

    return viz


# ===========================================================================
# Entry point: replay Block II checkpoints
# ===========================================================================

if __name__ == "__main__":
    block_ii_dir = _HERE / "block_ii"
    block_iii_dir = _HERE / "block_iii"

    # Try Block II checkpoints first
    if block_ii_dir.exists():
        print("=" * 50)
        print("Terrain Viz: Block II checkpoints")
        print("=" * 50)

        # Need full_params_fn from Block II
        sys.path.insert(0, str(_HERE))
        try:
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            from block_ii.block_ii import full_params as b2_full_params
            import jax.numpy as jnp

            def b2_full_np(params):
                p = {k: jnp.array(v) for k, v in params.items()}
                w, mu, sigma = b2_full_params(p)
                return np.asarray(w), np.asarray(mu), np.asarray(sigma)

            viz = replay_from_checkpoints(block_ii_dir, full_params_fn=b2_full_np)
        except Exception as e:
            print(f"  Error: {e}")
            # Fallback: try loading trained params directly
            trained = block_ii_dir / "block2_trained_params.npy"
            if trained.exists():
                data = np.load(str(trained), allow_pickle=True).item()
                if 'w_full' in data:
                    snapshot_3d(data['w_full'], data['mu_full'],
                                data['sigma_full'],
                                output_path=str(block_ii_dir / "terrain_3d.png"),
                                title="Block II Trained Potential")

    # Also check Block III
    if block_iii_dir.exists():
        npy_files = list(block_iii_dir.glob("*params*.npy"))
        if npy_files:
            print("\n" + "=" * 50)
            print("Terrain Viz: Block III checkpoints")
            print("=" * 50)
            replay_from_checkpoints(block_iii_dir)

    print("\nDone.")
