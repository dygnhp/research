"""
Control group: a standard TensorFlow Sequential ANN classifying the SAME
OX / ABC / abcd glyph images as CHM.

This is a black-box baseline for the white-box Contact Hamiltonian Machine.
It uses the identical datasets (via FINAL.data.load_dataset, same seeds) so
the accuracy is directly comparable to CHM's results in EXPERIMENT_RESULTS.md.
FLOPs / compute comparison is intentionally deferred (separate later step).

Run (after `pip install tensorflow`; CPU is fine), from the repo root:
    python -m control.ann_baseline                         # all three datasets
    python -m control.ann_baseline --datasets OX_8 --hidden 32 --epochs 150

Reported per dataset:
  - parameter count (vs CHM, which is parameter-light: it works in the lifted
    particle space, not on raw pixels)
  - canonical accuracy            (the C reference glyphs)
  - training-variant accuracy     (the same variant set CHM trained on)
  - held-out accuracy             (fresh variants, unseen seed -> generalization)

NB on fairness: CHM and the ANN solve the same *task* (classify the glyph) with
different *inputs* -- CHM lifts on-pixels to particles, the ANN flattens the
raw image. These synthetic glyphs are easy, so a small MLP reaches high
accuracy; the research point is the interpretability (CHM white-box vs ANN
black-box) and the compute trade-off, not accuracy superiority.
"""
from __future__ import annotations

import argparse
import os
import numpy as np

# Keep everything on CPU and quiet (the control runs after the CHM experiment;
# no GPU contention, and we only need a tiny MLP).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ---------------------------------------------------------------------------
# Data (reuses the exact CHM datasets)
# ---------------------------------------------------------------------------
def _labels(name):
    from FINAL.data import DATASETS
    return list(DATASETS[name].class_labels)


def _load_xy(name, n_per_class, seed):
    """Flattened images + integer labels for one dataset/seed."""
    from FINAL.data import load_dataset
    data = load_dataset(name, n_per_class=n_per_class, seed=seed)
    labels = _labels(name)
    idx = {lab: i for i, lab in enumerate(labels)}
    X, y = [], []
    for lab in labels:
        for img in data["images_by_label"][lab]:
            X.append(np.asarray(img, dtype=np.float32).ravel())
            y.append(idx[lab])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def _canonical_xy(name):
    from FINAL.data import DATASETS
    spec = DATASETS[name]
    labels = list(spec.class_labels)
    X = np.asarray([np.asarray(spec.canonicals[lab], np.float32).ravel()
                    for lab in labels], dtype=np.float32)
    y = np.arange(len(labels), dtype=np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(input_dim, n_classes, hidden):
    from tensorflow import keras
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(hidden, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ], name="ann_control")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def run_dataset(name, hidden=32, epochs=150, seed=42, n_per_class=50):
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)

    Xtr, ytr = _load_xy(name, n_per_class, seed)            # CHM's training data
    Xcanon, ycanon = _canonical_xy(name)
    Xho, yho = _load_xy(name, n_per_class, seed + 10_000)   # held-out fresh variants

    model = build_model(Xtr.shape[1], int(ytr.max()) + 1, hidden)
    model.fit(Xtr, ytr, epochs=epochs, batch_size=16, verbose=0)

    acc = lambda X, y: float(model.evaluate(X, y, verbose=0)[1])
    return {
        "dataset": name,
        "hidden": hidden,
        "params": int(model.count_params()),
        "canonical": acc(Xcanon, ycanon),
        "train_variant": acc(Xtr, ytr),
        "heldout": acc(Xho, yho),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# CHM reference (main_exp_2, improved + learnable sigma, full budget, 3-seed
# mean) for an at-a-glance contrast.  ANN compute/FLOPs comparison is deferred.
_CHM_REF = {
    "OX_8":    {"canonical": 1.00, "variant": 0.863, "params": "~434 (K54,D6)"},
    "ABC_16":  {"canonical": 1.00, "variant": 0.933, "params": "~262 (K37,D5)"},
    "abcd_32": {"canonical": 1.00, "variant": 0.708, "params": "~470 (K58,D6)"},
}


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(prog="control.ann_baseline")
    parser.add_argument("--datasets", default="OX_8,ABC_16,abcd_32",
                        help="comma-separated dataset names")
    parser.add_argument("--hidden", type=int, default=32,
                        help="hidden units in the single Dense layer")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seeds", default="42,1,2,3,4",
                        help="comma-separated seeds (one ANN run each)")
    parser.add_argument("--out", default=None,
                        help="optional JSON path for the aggregated results")
    args = parser.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    agg = {}
    for name in datasets:
        runs = []
        for sd in seeds:
            r = run_dataset(name, hidden=args.hidden, epochs=args.epochs,
                            seed=sd)
            runs.append(r)
            print(f"[ANN] {name:8s} seed={sd:<3d} params={r['params']:6d} | "
                  f"canonical={r['canonical']:.2f} "
                  f"train_var={r['train_variant']:.2f} "
                  f"heldout={r['heldout']:.2f}", flush=True)

        def ms(key):
            v = np.array([r[key] for r in runs], dtype=float)
            return float(v.mean()), float(v.std())
        agg[name] = {
            "params": runs[0]["params"], "hidden": args.hidden,
            "n_seeds": len(seeds),
            "canonical_mean_std": ms("canonical"),
            "train_variant_mean_std": ms("train_variant"),
            "heldout_mean_std": ms("heldout"),
            "heldout_per_seed": [r["heldout"] for r in runs],
        }

    print(f"\n=== ANN control: {len(seeds)} seeds {seeds} (mean+-std) "
          f"vs CHM (main_exp_2) ===")
    print(f"{'dataset':9s} | {'ANN params':10s} {'ANN canon':14s} "
          f"{'ANN heldout':16s} | {'CHM params':14s} {'CHM variant':11s}")
    for name, a in agg.items():
        cm, cs = a["canonical_mean_std"]
        hm, hs = a["heldout_mean_std"]
        c = _CHM_REF.get(name, {})
        print(f"{name:9s} | {a['params']:<10d} {cm:.3f}+-{cs:.3f}    "
              f"{hm:.3f}+-{hs:.3f}     | {str(c.get('params','?')):14s} "
              f"{str(c.get('variant','?')):11s}")
    print("\n주: ANN=black-box(가중치 의미 없음), CHM=white-box(지형/궤적 해석가능). "
          "연산량(FLOPs) 비교는 추후.")

    out_path = args.out or os.path.join("research", "ann_control.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump({"seeds": seeds, "hidden": args.hidden, "epochs": args.epochs,
               "results": agg}, open(out_path, "w"), indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
