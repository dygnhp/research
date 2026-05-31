"""
=============================================================================
O/X Image Dataset Generator
=============================================================================
Generates parametric 8x8 binary images of O (ring) and X (diagonal cross)
patterns with controlled variation for dataset-driven training.

O pattern: ring/ellipse defined by (center, inner_radius, outer_radius)
  - Pixels lie in annular region -> axis-connected neighbors
  - Contextual z ~ 0.88 (high) after preprocessing

X pattern: diagonal cross defined by (center, thickness)
  - Pixels lie near two diagonals -> diagonally-connected neighbors
  - Contextual z ~ 0.12 (low) after preprocessing

This structural difference in local connectivity is what the contact
Hamiltonian classifier exploits via the z-channel.

Usage:
    from data_generator import generate_dataset, O_CANONICAL, X_CANONICAL
    dataset = generate_dataset(n_per_class=50, seed=42)
=============================================================================
"""

import numpy as np


# ===========================================================================
# Canonical images (identical to Block I/II hardcoded versions)
# ===========================================================================
O_CANONICAL = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,0,0],
    [0,1,0,0,0,0,1,0],
    [0,1,0,0,0,0,1,0],
    [0,1,0,0,0,0,1,0],
    [0,1,0,0,0,0,1,0],
    [0,0,1,1,1,1,0,0],
    [0,0,0,0,0,0,0,0],
], dtype=float)

X_CANONICAL = np.array([
    [1,0,0,0,0,0,0,1],
    [0,1,0,0,0,0,1,0],
    [0,0,1,0,0,1,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,1,0,0,1,0,0],
    [0,1,0,0,0,0,1,0],
    [1,0,0,0,0,0,0,1],
], dtype=float)


# ===========================================================================
# Parametric generators
# ===========================================================================
def generate_O(cx=3.5, cy=3.5, r_inner=2.3, r_outer=3.3,
               aspect=1.0, noise_prob=0.0, rng=None):
    """
    Generate an O (ring/ellipse) pattern on an 8x8 grid.

    Args:
        cx, cy      : center in image coordinates (row=cy, col=cx convention)
        r_inner     : inner radius of the ring
        r_outer     : outer radius of the ring
        aspect      : aspect ratio (>1 = wider, <1 = taller)
        noise_prob  : probability of flipping each pixel
        rng         : numpy RandomState (for reproducibility)
    Returns:
        8x8 numpy array (float, 0.0 or 1.0)
    """
    img = np.zeros((8, 8), dtype=float)
    for r in range(8):
        for c in range(8):
            dx = (c - cx) / max(aspect, 0.3)
            dy = (r - cy) * max(aspect, 0.3) if aspect < 1.0 else (r - cy)
            if aspect >= 1.0:
                dx = (c - cx) / aspect
                dy = (r - cy)
            else:
                dx = (c - cx)
                dy = (r - cy) / aspect
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if r_inner <= dist <= r_outer:
                img[r, c] = 1.0

    if noise_prob > 0 and rng is not None:
        flip = rng.random((8, 8)) < noise_prob
        img = np.abs(img - flip.astype(float))

    return img


def generate_X(cx=3.5, cy=3.5, thickness=0.6, arm_scale=1.0,
               noise_prob=0.0, rng=None):
    """
    Generate an X (diagonal cross) pattern on an 8x8 grid.

    Args:
        cx, cy      : center in image coordinates
        thickness   : half-width of each diagonal arm
        arm_scale   : scale factor for arm length (1.0 = full corners)
        noise_prob  : probability of flipping each pixel
        rng         : numpy RandomState
    Returns:
        8x8 numpy array (float, 0.0 or 1.0)
    """
    img = np.zeros((8, 8), dtype=float)
    # Max radial distance from center to grid corner is ~4.95;
    # arm_scale=1.0 should reach corners.
    max_arm = 5.0 * arm_scale
    for r in range(8):
        for c in range(8):
            dr = r - cy
            dc = c - cx
            # Distance to main diagonal (dr = dc line)
            d1 = abs(dr - dc) / np.sqrt(2)
            # Distance to anti-diagonal (dr = -dc line)
            d2 = abs(dr + dc) / np.sqrt(2)
            radial = np.sqrt(dr ** 2 + dc ** 2)
            if radial <= max_arm and min(d1, d2) <= thickness:
                img[r, c] = 1.0

    if noise_prob > 0 and rng is not None:
        flip = rng.random((8, 8)) < noise_prob
        img = np.abs(img - flip.astype(float))

    return img


# ===========================================================================
# Random variant generators
# ===========================================================================
def generate_random_O(rng):
    """
    Generate a random O variant with randomized parameters.
    Ensures at least 8 ON pixels for meaningful physics.
    """
    for _ in range(20):
        cx = 3.5 + rng.uniform(-0.8, 0.8)
        cy = 3.5 + rng.uniform(-0.8, 0.8)
        r_inner = rng.uniform(1.5, 2.5)
        r_outer = r_inner + rng.uniform(0.7, 1.3)
        aspect = rng.uniform(0.8, 1.25)
        noise_p = rng.choice([0.0, 0.0, 0.0, 0.02, 0.05])
        img = generate_O(cx, cy, r_inner, r_outer, aspect, noise_p, rng)
        if img.sum() >= 8:
            return img
    # Fallback: canonical
    return O_CANONICAL.copy()


def generate_random_X(rng):
    """
    Generate a random X variant with randomized parameters.
    Ensures at least 8 ON pixels for meaningful physics.
    """
    for _ in range(20):
        cx = 3.5 + rng.uniform(-0.8, 0.8)
        cy = 3.5 + rng.uniform(-0.8, 0.8)
        thickness = rng.uniform(0.4, 0.85)
        arm_scale = rng.uniform(0.8, 1.0)
        noise_p = rng.choice([0.0, 0.0, 0.0, 0.02, 0.05])
        img = generate_X(cx, cy, thickness, arm_scale, noise_p, rng)
        if img.sum() >= 8:
            return img
    return X_CANONICAL.copy()


# ===========================================================================
# Dataset generation
# ===========================================================================
def generate_dataset(n_per_class=50, seed=42):
    """
    Generate a balanced dataset of O and X image variants.

    The first image in each class is always the canonical version
    (for backward compatibility and validation).

    Args:
        n_per_class : number of images per class
        seed        : random seed for reproducibility
    Returns:
        dict with keys:
            'images'  : list of 8x8 numpy arrays
            'labels'  : list of 'O' or 'X' strings
            'O_images': list of O images only
            'X_images': list of X images only
    """
    rng = np.random.RandomState(seed)

    O_images = [O_CANONICAL.copy()]
    for _ in range(n_per_class - 1):
        O_images.append(generate_random_O(rng))

    X_images = [X_CANONICAL.copy()]
    for _ in range(n_per_class - 1):
        X_images.append(generate_random_X(rng))

    images = O_images + X_images
    labels = ['O'] * n_per_class + ['X'] * n_per_class

    return {
        'images':   images,
        'labels':   labels,
        'O_images': O_images,
        'X_images': X_images,
        'n_per_class': n_per_class,
        'seed': seed,
    }


# ===========================================================================
# Visualization utility
# ===========================================================================
def visualize_dataset(dataset, n_show=8, output_path=None):
    """Show a grid of sample O and X images from the dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(n_show, dataset['n_per_class'])
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.5, 3.5))

    for i in range(n_show):
        axes[0, i].imshow(dataset['O_images'][i], cmap='Blues',
                          vmin=0, vmax=1, interpolation='nearest')
        axes[0, i].set_title(f'O #{i}', fontsize=8)
        axes[0, i].axis('off')
        n_pix = int(dataset['O_images'][i].sum())
        axes[0, i].text(0.5, -0.1, f'{n_pix}px', transform=axes[0, i].transAxes,
                        ha='center', fontsize=7, color='gray')

        axes[1, i].imshow(dataset['X_images'][i], cmap='Reds',
                          vmin=0, vmax=1, interpolation='nearest')
        axes[1, i].set_title(f'X #{i}', fontsize=8)
        axes[1, i].axis('off')
        n_pix = int(dataset['X_images'][i].sum())
        axes[1, i].text(0.5, -0.1, f'{n_pix}px', transform=axes[1, i].transAxes,
                        ha='center', fontsize=7, color='gray')

    fig.suptitle(f'Dataset Sample (seed={dataset["seed"]}, '
                 f'{dataset["n_per_class"]}/class)', fontsize=10)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        print(f"  [Dataset figure] -> {output_path}")
    plt.close(fig)


# ===========================================================================
# Self-test
# ===========================================================================
if __name__ == "__main__":
    print("Testing O/X image generator...")

    # Test canonical images match
    o_gen = generate_O()
    x_gen = generate_X()
    o_match = np.array_equal(o_gen, O_CANONICAL)
    x_match = np.array_equal(x_gen, X_CANONICAL)
    print(f"  O canonical match: {o_match}  ({int(o_gen.sum())} px)")
    print(f"  X canonical match: {x_match}  ({int(x_gen.sum())} px)")

    if not o_match:
        print("  O diff positions:")
        for r in range(8):
            for c in range(8):
                if o_gen[r, c] != O_CANONICAL[r, c]:
                    print(f"    ({r},{c}): gen={o_gen[r,c]:.0f} canon={O_CANONICAL[r,c]:.0f}")

    if not x_match:
        print("  X diff positions:")
        for r in range(8):
            for c in range(8):
                if x_gen[r, c] != X_CANONICAL[r, c]:
                    print(f"    ({r},{c}): gen={x_gen[r,c]:.0f} canon={X_CANONICAL[r,c]:.0f}")

    # Test dataset generation
    ds = generate_dataset(n_per_class=50, seed=42)
    print(f"\n  Dataset: {len(ds['images'])} images "
          f"({ds['n_per_class']} O + {ds['n_per_class']} X)")

    pix_O = [int(img.sum()) for img in ds['O_images']]
    pix_X = [int(img.sum()) for img in ds['X_images']]
    print(f"  O pixel counts: min={min(pix_O)} max={max(pix_O)} "
          f"mean={np.mean(pix_O):.1f}")
    print(f"  X pixel counts: min={min(pix_X)} max={max(pix_X)} "
          f"mean={np.mean(pix_X):.1f}")

    # Save visualization (skip if matplotlib not available)
    try:
        visualize_dataset(ds, n_show=10, output_path="dataset_sample.png")
    except ImportError:
        print("  (matplotlib not available, skipping visualization)")
    print("\nDone.")
