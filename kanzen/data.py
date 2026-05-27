"""
Image dataset registry.

Three datasets are bundled, each with its own canonical images, parametric
variant generator, and N-class attractor layout.  The Config object
selects one of them by name.

    OX_8     8x8   2 classes  O, X            (rings vs diagonal crosses)
    ABC_16   16x16 3 classes  A, B, C         (block-letter uppercase)
    abcd_32  32x32 4 classes  a, b, c, d      (printed-style lowercase)

Each class is defined by:
  - a canonical image (parametric default arguments produce it exactly),
  - a 'random' generator (uniform-jitter on the parametric arguments),
  - an attractor position in the lifted (x, y) plane,
  - a class-typical z value that anchors the attractor in the connectivity
    dimension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable
import numpy as np


# ===========================================================================
# Drawing primitives (used by the parametric generators below)
# ===========================================================================
def _seg_dist(px, py, ax, ay, bx, by):
    """Distance from (px, py) to segment (ax, ay) - (bx, by)."""
    abx, aby = bx - ax, by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq < 1e-9:
        return np.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_sq))
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))


def _draw_line(img, p0, p1, thickness):
    """Rasterize a thick line segment onto img."""
    H, W = img.shape
    half = thickness / 2.0
    for r in range(H):
        for c in range(W):
            if _seg_dist(c, r, p0[0], p0[1], p1[0], p1[1]) <= half:
                img[r, c] = 1.0


def _draw_arc(img, center, r_outer, r_inner, a_start_deg, a_end_deg):
    """Annular sector.  Full ring when (a_end - a_start) >= 360."""
    H, W = img.shape
    cx, cy = center
    full = (a_end_deg - a_start_deg) >= 360.0 - 1e-6
    a_s = np.radians(a_start_deg) % (2 * np.pi)
    a_e = np.radians(a_end_deg) % (2 * np.pi)
    for r in range(H):
        for c in range(W):
            dx, dy = c - cx, r - cy
            d = np.hypot(dx, dy)
            if not (r_inner <= d <= r_outer):
                continue
            if full:
                img[r, c] = 1.0
                continue
            ang = np.arctan2(dy, dx) % (2 * np.pi)
            ok = (a_s <= ang <= a_e) if a_s <= a_e else (ang >= a_s or ang <= a_e)
            if ok:
                img[r, c] = 1.0


def _add_noise(img, noise_prob, rng):
    if noise_prob > 0 and rng is not None:
        flip = rng.random(img.shape) < noise_prob
        img = np.abs(img - flip.astype(np.float32))
    return img


# ===========================================================================
# Connectivity feature (used to compute attractor z-coords)
# ===========================================================================
def _safe_px(img, r, c):
    rows, cols = img.shape
    if 0 <= r < rows and 0 <= c < cols:
        return float(img[r, c])
    return 0.0


def _mean_z(image, tau=0.5, beta=1.0):
    """Mean z-connectivity score across on-pixels.

    Same formula as preprocess.preprocess: a higher mean z means the pixels
    are mostly axis-connected (ring-like); a lower mean z means they are
    diagonally connected (cross-like).
    """
    rows, cols = image.shape
    vals = []
    for r in range(rows):
        for c in range(cols):
            if image[r, c] > tau:
                d_axis = (_safe_px(image, r - 1, c) + _safe_px(image, r + 1, c)
                          + _safe_px(image, r, c - 1) + _safe_px(image, r, c + 1))
                d_diag_signed = (_safe_px(image, r - 1, c - 1)
                                 + _safe_px(image, r + 1, c + 1)
                                 - _safe_px(image, r - 1, c + 1)
                                 - _safe_px(image, r + 1, c - 1))
                score = d_axis - abs(d_diag_signed)
                vals.append(1.0 / (1.0 + np.exp(-beta * score)))
    return float(np.mean(vals)) if vals else 0.5


# ===========================================================================
# OX_8: 8x8 O / X (original dataset)
# ===========================================================================
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
               aspect=1.0, noise_prob=0.0, rng=None):
    img = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            if aspect >= 1.0:
                dx, dy = (c - cx) / aspect, r - cy
            else:
                dx, dy = c - cx, (r - cy) / aspect
            dist = np.hypot(dx, dy)
            if r_inner <= dist <= r_outer:
                img[r, c] = 1.0
    return _add_noise(img, noise_prob, rng)


def generate_X(cx=3.5, cy=3.5, thickness=0.6, arm_scale=1.0,
               noise_prob=0.0, rng=None):
    img = np.zeros((8, 8), dtype=np.float32)
    max_arm = 5.0 * arm_scale
    for r in range(8):
        for c in range(8):
            dr, dc = r - cy, c - cx
            d1 = abs(dr - dc) / np.sqrt(2)
            d2 = abs(dr + dc) / np.sqrt(2)
            if np.hypot(dr, dc) <= max_arm and min(d1, d2) <= thickness:
                img[r, c] = 1.0
    return _add_noise(img, noise_prob, rng)


def generate_random_O(rng):
    for _ in range(20):
        cx = 3.5 + rng.uniform(-0.8, 0.8)
        cy = 3.5 + rng.uniform(-0.8, 0.8)
        r_in = rng.uniform(1.5, 2.5)
        r_out = r_in + rng.uniform(0.7, 1.3)
        aspect = rng.uniform(0.8, 1.25)
        n_p = rng.choice([0.0, 0.0, 0.0, 0.02, 0.05])
        img = generate_O(cx, cy, r_in, r_out, aspect, n_p, rng)
        if img.sum() >= 8:
            return img
    return O_CANONICAL.copy()


def generate_random_X(rng):
    for _ in range(20):
        cx = 3.5 + rng.uniform(-0.8, 0.8)
        cy = 3.5 + rng.uniform(-0.8, 0.8)
        th = rng.uniform(0.4, 0.85)
        arm = rng.uniform(0.8, 1.0)
        n_p = rng.choice([0.0, 0.0, 0.0, 0.02, 0.05])
        img = generate_X(cx, cy, th, arm, n_p, rng)
        if img.sum() >= 8:
            return img
    return X_CANONICAL.copy()


# ===========================================================================
# ABC_16: 16x16 A / B / C
# ===========================================================================
def generate_A_16(cx=7.5, cy=7.75, half_width=6.0, height=11.5,
                  thickness=1.6, cb_frac=0.55,
                  noise_prob=0.0, rng=None):
    img = np.zeros((16, 16), dtype=np.float32)
    top = (cx, cy - height / 2)
    bl = (cx - half_width, cy + height / 2)
    br = (cx + half_width, cy + height / 2)
    _draw_line(img, bl, top, thickness)
    _draw_line(img, br, top, thickness)
    cb_y = cy - height / 2 + height * cb_frac
    inset = half_width * (1 - cb_frac) * 0.7
    _draw_line(img, (cx - inset, cb_y), (cx + inset, cb_y), thickness * 0.9)
    return _add_noise(img, noise_prob, rng)


def generate_B_16(cx=4.5, cy=7.75, height=11.5, bowl_r_outer=3.5,
                  bowl_r_inner=2.2, thickness=1.8,
                  noise_prob=0.0, rng=None):
    img = np.zeros((16, 16), dtype=np.float32)
    top_y = cy - height / 2
    bot_y = cy + height / 2
    _draw_line(img, (cx, top_y), (cx, bot_y), thickness)
    top_arc_cy = top_y + (cy - top_y) / 2 + 0.3
    bot_arc_cy = cy + (bot_y - cy) / 2 - 0.3
    _draw_arc(img, (cx + 0.5, top_arc_cy), bowl_r_outer, bowl_r_inner, -90, 90)
    _draw_arc(img, (cx + 0.5, bot_arc_cy), bowl_r_outer, bowl_r_inner, -90, 90)
    return _add_noise(img, noise_prob, rng)


def generate_C_16(cx=8.0, cy=8.0, r_outer=6.0, r_inner=4.4,
                  open_half_deg=50.0, noise_prob=0.0, rng=None):
    img = np.zeros((16, 16), dtype=np.float32)
    _draw_arc(img, (cx, cy), r_outer, r_inner,
              open_half_deg, 360.0 - open_half_deg)
    return _add_noise(img, noise_prob, rng)


def generate_random_A_16(rng):
    for _ in range(20):
        cx = 7.5 + rng.uniform(-0.8, 0.8)
        cy = 7.75 + rng.uniform(-0.6, 0.6)
        hw = rng.uniform(5.4, 6.4)
        h = rng.uniform(10.5, 12.0)
        th = rng.uniform(1.5, 1.8)
        cb_f = rng.uniform(0.5, 0.6)
        n_p = rng.choice([0.0, 0.0, 0.02, 0.04])
        img = generate_A_16(cx, cy, hw, h, th, cb_f, n_p, rng)
        if img.sum() >= 30:
            return img
    return A_CANONICAL_16.copy()


def generate_random_B_16(rng):
    for _ in range(20):
        cx = 4.5 + rng.uniform(-0.6, 0.6)
        cy = 7.75 + rng.uniform(-0.6, 0.6)
        h = rng.uniform(10.5, 12.0)
        ro = rng.uniform(3.3, 3.7)
        ri = ro - rng.uniform(1.1, 1.5)
        th = rng.uniform(1.6, 2.0)
        n_p = rng.choice([0.0, 0.0, 0.02, 0.04])
        img = generate_B_16(cx, cy, h, ro, ri, th, n_p, rng)
        if img.sum() >= 30:
            return img
    return B_CANONICAL_16.copy()


def generate_random_C_16(rng):
    for _ in range(20):
        cx = 8.0 + rng.uniform(-0.8, 0.8)
        cy = 8.0 + rng.uniform(-0.8, 0.8)
        ro = rng.uniform(5.6, 6.3)
        ri = ro - rng.uniform(1.4, 1.8)
        oh = rng.uniform(40.0, 55.0)
        n_p = rng.choice([0.0, 0.0, 0.02, 0.04])
        img = generate_C_16(cx, cy, ro, ri, oh, n_p, rng)
        if img.sum() >= 25:
            return img
    return C_CANONICAL_16.copy()


A_CANONICAL_16 = generate_A_16()
B_CANONICAL_16 = generate_B_16()
C_CANONICAL_16 = generate_C_16()


# ===========================================================================
# abcd_32: 32x32 a / b / c / d
# ===========================================================================
def generate_a_32(loop_cx=13.0, loop_cy=19.0, loop_r_outer=7.0,
                  loop_r_inner=4.5, bar_x=20.5, bar_top=11.0,
                  bar_bot=25.5, bar_thickness=2.4,
                  noise_prob=0.0, rng=None):
    img = np.zeros((32, 32), dtype=np.float32)
    _draw_arc(img, (loop_cx, loop_cy), loop_r_outer, loop_r_inner, 0, 360)
    _draw_line(img, (bar_x, bar_top), (bar_x, bar_bot), bar_thickness)
    return _add_noise(img, noise_prob, rng)


def generate_b_32(bar_x=10.5, bar_top=4.0, bar_bot=27.0,
                  bowl_cx=15.5, bowl_cy=19.0, bowl_r_outer=7.0,
                  bowl_r_inner=4.5, bar_thickness=2.4,
                  noise_prob=0.0, rng=None):
    img = np.zeros((32, 32), dtype=np.float32)
    _draw_line(img, (bar_x, bar_top), (bar_x, bar_bot), bar_thickness)
    _draw_arc(img, (bowl_cx, bowl_cy), bowl_r_outer, bowl_r_inner, 0, 360)
    return _add_noise(img, noise_prob, rng)


def generate_c_32(cx=16.0, cy=16.0, r_outer=10.0, r_inner=7.5,
                  open_half_deg=55.0, noise_prob=0.0, rng=None):
    img = np.zeros((32, 32), dtype=np.float32)
    _draw_arc(img, (cx, cy), r_outer, r_inner,
              open_half_deg, 360.0 - open_half_deg)
    return _add_noise(img, noise_prob, rng)


def generate_d_32(bar_x=21.5, bar_top=4.0, bar_bot=27.0,
                  bowl_cx=16.5, bowl_cy=19.0, bowl_r_outer=7.0,
                  bowl_r_inner=4.5, bar_thickness=2.4,
                  noise_prob=0.0, rng=None):
    img = np.zeros((32, 32), dtype=np.float32)
    _draw_line(img, (bar_x, bar_top), (bar_x, bar_bot), bar_thickness)
    _draw_arc(img, (bowl_cx, bowl_cy), bowl_r_outer, bowl_r_inner, 0, 360)
    return _add_noise(img, noise_prob, rng)


def generate_random_a_32(rng):
    for _ in range(20):
        lcx = 13.0 + rng.uniform(-1.0, 1.0)
        lcy = 19.0 + rng.uniform(-1.0, 1.0)
        ro = rng.uniform(6.5, 7.5)
        ri = ro - rng.uniform(2.2, 2.8)
        bx = 20.5 + rng.uniform(-0.5, 0.5)
        th = rng.uniform(2.2, 2.6)
        n_p = rng.choice([0.0, 0.0, 0.01, 0.02])
        img = generate_a_32(lcx, lcy, ro, ri, bx, 11.0, 25.5, th, n_p, rng)
        if img.sum() >= 80:
            return img
    return a_CANONICAL_32.copy()


def generate_random_b_32(rng):
    for _ in range(20):
        bx = 10.5 + rng.uniform(-0.8, 0.8)
        cx = 15.5 + rng.uniform(-1.0, 1.0)
        cy = 19.0 + rng.uniform(-1.0, 1.0)
        ro = rng.uniform(6.5, 7.5)
        ri = ro - rng.uniform(2.2, 2.8)
        th = rng.uniform(2.2, 2.6)
        n_p = rng.choice([0.0, 0.0, 0.01, 0.02])
        img = generate_b_32(bx, 4.0, 27.0, cx, cy, ro, ri, th, n_p, rng)
        if img.sum() >= 80:
            return img
    return b_CANONICAL_32.copy()


def generate_random_c_32(rng):
    for _ in range(20):
        cx = 16.0 + rng.uniform(-1.0, 1.0)
        cy = 16.0 + rng.uniform(-1.0, 1.0)
        ro = rng.uniform(9.5, 10.5)
        ri = ro - rng.uniform(2.2, 2.8)
        oh = rng.uniform(48.0, 60.0)
        n_p = rng.choice([0.0, 0.0, 0.01, 0.02])
        img = generate_c_32(cx, cy, ro, ri, oh, n_p, rng)
        if img.sum() >= 60:
            return img
    return c_CANONICAL_32.copy()


def generate_random_d_32(rng):
    for _ in range(20):
        bx = 21.5 + rng.uniform(-0.8, 0.8)
        cx = 16.5 + rng.uniform(-1.0, 1.0)
        cy = 19.0 + rng.uniform(-1.0, 1.0)
        ro = rng.uniform(6.5, 7.5)
        ri = ro - rng.uniform(2.2, 2.8)
        th = rng.uniform(2.2, 2.6)
        n_p = rng.choice([0.0, 0.0, 0.01, 0.02])
        img = generate_d_32(bx, 4.0, 27.0, cx, cy, ro, ri, th, n_p, rng)
        if img.sum() >= 80:
            return img
    return d_CANONICAL_32.copy()


a_CANONICAL_32 = generate_a_32()
b_CANONICAL_32 = generate_b_32()
c_CANONICAL_32 = generate_c_32()
d_CANONICAL_32 = generate_d_32()


# ===========================================================================
# Dataset registry
# ===========================================================================
@dataclass
class DatasetSpec:
    name: str
    image_size: Tuple[int, int]        # (H, W)
    class_labels: List[str]
    canonicals: Dict[str, np.ndarray]
    random_generators: Dict[str, Callable]
    attractor_positions: Dict[str, Tuple[float, float]]
    attractor_z: Dict[str, float]
    n_max: int
    tau: float = 0.5

    @property
    def n_classes(self) -> int:
        return len(self.class_labels)


def _polygon_attractors(n: int, radius: float, center: Tuple[float, float],
                        labels: List[str], start_deg: float = 0.0
                        ) -> Dict[str, Tuple[float, float]]:
    """Place N attractors on a circle of given radius around 'center',
    starting at 'start_deg' and going CCW with uniform spacing."""
    out = {}
    for i, lab in enumerate(labels):
        ang = np.radians(start_deg + 360.0 * i / n)
        out[lab] = (center[0] + radius * np.cos(ang),
                    center[1] + radius * np.sin(ang))
    return out


DATASETS: Dict[str, DatasetSpec] = {
    "OX_8": DatasetSpec(
        name="OX_8",
        image_size=(8, 8),
        class_labels=["O", "X"],
        canonicals={"O": O_CANONICAL, "X": X_CANONICAL},
        random_generators={"O": generate_random_O, "X": generate_random_X},
        # Phase A revision: attractor positions are now SYMMETRIC around the
        # data center (3.5, 3.5).  The original (+/-8, +/-8) placement was
        # asymmetric -- O at distance 6.36 from center, X at distance 16.26 --
        # which made the X-attractor inert regardless of sigma.  With these
        # symmetric positions, both attractors lie equidistant from the data
        # center and a moderate sigma_frozen suffices to give actual gradient.
        attractor_positions={"O": (10.0, 10.0), "X": (-3.0, -3.0)},
        attractor_z={"O": 0.88, "X": 0.12},
        n_max=64,
        tau=0.5,
    ),
    "ABC_16": DatasetSpec(
        name="ABC_16",
        image_size=(16, 16),
        class_labels=["A", "B", "C"],
        canonicals={
            "A": A_CANONICAL_16,
            "B": B_CANONICAL_16,
            "C": C_CANONICAL_16,
        },
        random_generators={
            "A": generate_random_A_16,
            "B": generate_random_B_16,
            "C": generate_random_C_16,
        },
        # Phase A: triangle of radius 14 centered on (7.5, 7.5)
        # (was 20; closer attractors give actual gradient at data domain edges)
        attractor_positions=_polygon_attractors(
            3, 14.0, (7.5, 7.5), ["A", "B", "C"], start_deg=90.0),
        attractor_z={
            "A": _mean_z(A_CANONICAL_16),
            "B": _mean_z(B_CANONICAL_16),
            "C": _mean_z(C_CANONICAL_16),
        },
        n_max=128,
        tau=0.5,
    ),
    "abcd_32": DatasetSpec(
        name="abcd_32",
        image_size=(32, 32),
        class_labels=["a", "b", "c", "d"],
        canonicals={
            "a": a_CANONICAL_32,
            "b": b_CANONICAL_32,
            "c": c_CANONICAL_32,
            "d": d_CANONICAL_32,
        },
        random_generators={
            "a": generate_random_a_32,
            "b": generate_random_b_32,
            "c": generate_random_c_32,
            "d": generate_random_d_32,
        },
        # Phase A: square of radius 26 centered on (15.5, 15.5)
        # (was 36; reduced for the same reason as ABC_16)
        attractor_positions=_polygon_attractors(
            4, 26.0, (15.5, 15.5), ["a", "b", "c", "d"], start_deg=45.0),
        attractor_z={
            "a": _mean_z(a_CANONICAL_32),
            "b": _mean_z(b_CANONICAL_32),
            "c": _mean_z(c_CANONICAL_32),
            "d": _mean_z(d_CANONICAL_32),
        },
        n_max=400,
        tau=0.5,
    ),
}


# ===========================================================================
# Loader
# ===========================================================================
def load_dataset(name: str = "OX_8", n_per_class: int = 50,
                 seed: int = 42) -> Dict:
    """Build a dataset from the named registry entry.

    The first image of each class is the canonical one (deterministic);
    the remaining n_per_class - 1 are sampled by the random variant
    generator.

    Returns:
        {
            "spec":             DatasetSpec,
            "images_by_label":  {label: [img, img, ...]},
            "n_per_class":      int,
            "seed":             int,
        }
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    spec = DATASETS[name]
    rng = np.random.RandomState(seed)
    images_by_label: Dict[str, List[np.ndarray]] = {}
    for label in spec.class_labels:
        imgs = [spec.canonicals[label].copy()]
        gen = spec.random_generators[label]
        for _ in range(n_per_class - 1):
            imgs.append(gen(rng))
        images_by_label[label] = imgs
    return {
        "spec": spec,
        "images_by_label": images_by_label,
        "n_per_class": n_per_class,
        "seed": seed,
    }


# Backward-compatible alias for the OX_8 dataset
def generate_dataset(n_per_class: int = 50, seed: int = 42) -> Dict:
    """OX_8 dataset in the legacy 2-class dict shape."""
    d = load_dataset("OX_8", n_per_class, seed)
    return {
        "O_images": d["images_by_label"]["O"],
        "X_images": d["images_by_label"]["X"],
        "n_per_class": d["n_per_class"],
        "seed": d["seed"],
    }
