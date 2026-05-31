"""
Dataset gallery generator (PART 1).

Render *every* training image of a dataset into a single numbered PNG so an
experimenter can eyeball exactly what the model is being trained on.  Each
class occupies its own block of rows; the first image of every class is the
canonical one (highlighted with a colored border), the rest are random
variants produced by the dataset's parametric generators.

This module is read-only with respect to kanzen.data: it calls
load_dataset() and never mutates any DatasetSpec or canonical array.

CLI:
    python -m kanzen.gallery --dataset OX_8 --n_per_class 50 --seed 42
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .data import load_dataset, DATASETS


# ---------------------------------------------------------------------------
# Layout heuristics
# ---------------------------------------------------------------------------
def _cols_for_image_size(side: int) -> int:
    """Pick a column count so small images pack wide and large images stay
    readable.  8x8 -> ~10 columns, 16x16 -> ~9, 32x32 -> ~8."""
    if side <= 8:
        return 10
    if side <= 16:
        return 9
    return 8


def _per_image_inch(side: int) -> float:
    """Physical size (inches) allotted to each cell, larger for bigger
    images so the pixel grid stays legible."""
    if side <= 8:
        return 1.1
    if side <= 16:
        return 1.3
    return 1.5


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------
def make_dataset_gallery(dataset_name: str,
                         n_per_class: int = 50,
                         seed: int = 42,
                         out_path: str = None) -> str:
    """Render all images of a dataset into one numbered PNG grid.

    Args:
        dataset_name : key in kanzen.data.DATASETS (e.g. "OX_8").
        n_per_class  : images per class (first is canonical, rest random).
        seed         : seed forwarded to load_dataset for reproducibility.
        out_path     : destination PNG.  When None, defaults to
                       research/experiment/datasets/{dataset_name}_gallery.png.

    Returns:
        The path the PNG was written to.
    """
    data = load_dataset(dataset_name, n_per_class=n_per_class, seed=seed)
    spec = data["spec"]
    images_by_label = data["images_by_label"]
    labels: List[str] = list(spec.class_labels)
    H, W = spec.image_size
    side = max(H, W)

    n_cols = _cols_for_image_size(side)

    # Each class gets a title row plus enough image rows to hold its images.
    # We lay every class out on its own set of rows so class boundaries
    # always force a line break (layout rule 1 & 3).
    rows_per_class = [int(np.ceil(len(images_by_label[lab]) / n_cols))
                      for lab in labels]
    title_rows = len(labels)                 # one title strip per class
    total_grid_rows = sum(rows_per_class) + title_rows

    cell = _per_image_inch(side)
    fig_w = n_cols * cell
    fig_h = total_grid_rows * cell + 0.6      # +0.6 for the suptitle band
    fig = plt.figure(figsize=(fig_w, fig_h))

    # A GridSpec with one logical row per (title strip | image row).
    gs = fig.add_gridspec(total_grid_rows, n_cols,
                          hspace=0.45, wspace=0.15)

    canonical_border = "red"
    global_idx = 1            # running #N across all classes, 1-based
    grid_row = 0

    for c, lab in enumerate(labels):
        imgs = images_by_label[lab]

        # ---- class title strip (spans the full width) -------------------
        ax_title = fig.add_subplot(gs[grid_row, :])
        ax_title.axis("off")
        ax_title.text(0.5, 0.5,
                      f"Class '{lab}'  ({len(imgs)} images)",
                      ha="center", va="center",
                      fontsize=14, fontweight="bold",
                      transform=ax_title.transAxes)
        grid_row += 1

        # ---- image cells -------------------------------------------------
        for i, img in enumerate(imgs):
            r = grid_row + i // n_cols
            col = i % n_cols
            ax = fig.add_subplot(gs[r, col])
            ax.imshow(img, cmap="gray_r", vmin=0.0, vmax=1.0,
                      interpolation="nearest")

            # Thin pixel-boundary grid so individual pixels are visible.
            ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
            ax.grid(which="minor", color="lightgray", linewidth=0.3)
            ax.set_xticks([])
            ax.set_yticks([])

            is_canonical = (i == 0)
            if is_canonical:
                # Highlight the canonical image with a colored frame.
                for spine in ax.spines.values():
                    spine.set_edgecolor(canonical_border)
                    spine.set_linewidth(2.2)
                tag = f"#{global_idx} ({lab})*"
                title_color = canonical_border
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor("0.6")
                    spine.set_linewidth(0.5)
                tag = f"#{global_idx} ({lab})"
                title_color = "black"

            ax.set_title(tag, fontsize=7, color=title_color, pad=2)
            global_idx += 1

        grid_row += rows_per_class[c]

    n_total = global_idx - 1
    fig.suptitle(
        f"{dataset_name}  |  {len(labels)} classes "
        f"({', '.join(labels)})  |  {n_per_class}/class  "
        f"|  {n_total} images  |  seed={seed}\n"
        f"(* = canonical, red border)",
        fontsize=12, y=0.995)

    # ---- save ------------------------------------------------------------
    if out_path is None:
        out_path = os.path.join("research", "experiment", "datasets",
                                f"{dataset_name}_gallery.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- report ----------------------------------------------------------
    print(f"[gallery] 데이터셋: {dataset_name}")
    print(f"[gallery] 클래스: {labels} ({len(labels)}개)")
    print(f"[gallery] 총 이미지: {n_total}장 (클래스당 {n_per_class}장)")
    print(f"[gallery] 저장 완료: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    # Accept both registry keys (OX_8) and the friendly aliases the runner
    # uses (OX / ABC / abcd) so the CLI is consistent across tools.
    aliases = {"OX": "OX_8", "ABC": "ABC_16", "abcd": "abcd_32"}
    choices = list(DATASETS.keys()) + list(aliases.keys())
    parser = argparse.ArgumentParser(prog="FINAL.gallery")
    parser.add_argument("--dataset", choices=choices, default="OX_8")
    parser.add_argument("--n_per_class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)
    dataset = aliases.get(args.dataset, args.dataset)
    make_dataset_gallery(dataset, n_per_class=args.n_per_class,
                         seed=args.seed, out_path=args.out)


if __name__ == "__main__":
    main()
