"""
Parametric 8x8 O/X image generator.

The O class is a ring/ellipse; the X class is a diagonal cross.  Each class
has a canonical image (deterministic) and a random-variant generator.  The
'generate_dataset' helper returns a balanced O/X dataset whose first entry
of each class is always canonical (for backward-compatible validation).
"""
from __future__ import annotations

import numpy as np


O_CANONICAL = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.float32)

X_CANONICAL = np.array([
    [1, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 1],
], dtype=np.float32)


def generate_O(cx=3.5, cy=3.5, r_inner=2.3, r_outer=3.3,
               aspect=1.0, noise_prob=0.0, rng=None) -> np.ndarray:
    img = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
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
        img = np.abs(img - flip.astype(np.float32))
    return img


def generate_X(cx=3.5, cy=3.5, thickness=0.6, arm_scale=1.0,
               noise_prob=0.0, rng=None) -> np.ndarray:
    img = np.zeros((8, 8), dtype=np.float32)
    max_arm = 5.0 * arm_scale
    for r in range(8):
        for c in range(8):
            dr, dc = r - cy, c - cx
            d1 = abs(dr - dc) / np.sqrt(2)
            d2 = abs(dr + dc) / np.sqrt(2)
            radial = np.sqrt(dr ** 2 + dc ** 2)
            if radial <= max_arm and min(d1, d2) <= thickness:
                img[r, c] = 1.0
    if noise_prob > 0 and rng is not None:
        flip = rng.random((8, 8)) < noise_prob
        img = np.abs(img - flip.astype(np.float32))
    return img


def generate_random_O(rng) -> np.ndarray:
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
    return O_CANONICAL.copy()


def generate_random_X(rng) -> np.ndarray:
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


def generate_dataset(n_per_class: int = 50, seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    O_images = [O_CANONICAL.copy()]
    for _ in range(n_per_class - 1):
        O_images.append(generate_random_O(rng))
    X_images = [X_CANONICAL.copy()]
    for _ in range(n_per_class - 1):
        X_images.append(generate_random_X(rng))
    return {
        "O_images": O_images,
        "X_images": X_images,
        "n_per_class": n_per_class,
        "seed": seed,
    }
