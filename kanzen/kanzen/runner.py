"""
CLI ready-to-experiment runner (replaces the planned tkinter GUI).

Pick four knobs -- dataset, epochs, initial K, initial D -- get a calibrated
wall-time estimate measured on *this* machine, confirm, and then conduct the
full experiment: dataset gallery -> training (with autonomous growth) ->
terrain time-evolution (3D) -> evaluation -> saved artifacts.

    python -m kanzen.runner --dataset OX --epochs 2000 --init-k 16 --init-d 3

Everything lands in research/experiment/exp_NNN/.

Design notes
------------
* Device flag is honored on a best-effort basis: JAX_PLATFORMS is set from a
  pre-scan of argv before the jax backend initializes.  GPU is only used if
  jax actually exposes a GPU device (native Windows jax is CPU-only; use WSL2
  for GPU).  The actually-active device is always reported via report_device.
* The time estimate comes from a live two-point calibration, never a
  hardcoded number, so it reflects this machine and this (K, D, dataset).
  It is a *training-step* estimate; periodic diagnostics and growth
  recompiles are not fully modeled, so a contingency upper bound is shown and
  the real elapsed time is printed at the end for comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Device env -- must run before the jax backend initializes
# ---------------------------------------------------------------------------
def _prescan_device() -> str:
    """Read --device from argv and pin JAX_PLATFORMS *before* jax inits.

    Returns the requested device string ('auto' / 'gpu' / 'cpu').
    """
    dev = "auto"
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            dev = argv[i + 1].lower()
        elif a.startswith("--device="):
            dev = a.split("=", 1)[1].lower()
    if dev == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    # For 'gpu'/'auto' we let jax choose; we never force a GPU that is absent.
    return dev


_REQUESTED_DEVICE = _prescan_device()

# Friendly dataset aliases -> registry keys.
_DATASET_ALIASES = {
    "OX": "OX_8", "OX_8": "OX_8",
    "ABC": "ABC_16", "ABC_16": "ABC_16",
    "abcd": "abcd_32", "abcd_32": "abcd_32",
}


def _improved_attractor_preset(dataset_key: str) -> dict:
    """Config overrides for the 'improved' attractor layout (B bundle).

    Brings each attractor into ~1.5-2 sigma of the particle cloud and makes
    the layout symmetric about the data center, so the frozen-attractor
    gradient actually reaches the data domain.  Returns Config kwargs:
    attractor_override (positions), attractor_sigma_init, frozen_w.
    Only the layout / sigma-init / depth change here; sigma itself is still
    learned (or frozen) per the --learn-sigma flag.
    """
    from .data import _polygon_attractors, DATASETS
    spec = DATASETS[dataset_key]
    labels = list(spec.class_labels)
    H, W = spec.image_size
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    if dataset_key == "OX_8":
        override = {"O": (8.0, 8.0), "X": (-1.0, -1.0)}   # symmetric about (3.5,3.5)
        sigma_init = 4.0
    elif dataset_key == "ABC_16":
        override = _polygon_attractors(3, 10.0, (cx, cy), labels, start_deg=90.0)
        sigma_init = 3.0
    elif dataset_key == "abcd_32":
        override = _polygon_attractors(4, 18.0, (cx, cy), labels, start_deg=45.0)
        sigma_init = 3.0
    else:
        return {}
    return {"attractor_override": override,
            "attractor_sigma_init": sigma_init,
            "frozen_w": -3.0}


# ---------------------------------------------------------------------------
# Experiment directory allocation
# ---------------------------------------------------------------------------
def _next_exp_dir(base: str = None) -> Tuple[str, int]:
    if base is None:
        base = os.path.join("research", "experiment")
    os.makedirs(base, exist_ok=True)
    nums = [int(m.group(1))
            for d in os.listdir(base)
            if (m := re.fullmatch(r"exp_(\d+)", d))]
    n = (max(nums) + 1) if nums else 1
    path = os.path.join(base, f"exp_{n:03d}")
    os.makedirs(path, exist_ok=True)
    return path, n


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(cfg, init_k: int, init_d: int, epochs: int) -> List[str]:
    """Return a list of human-readable validation errors (empty if OK)."""
    errors = []
    c = cfg.n_classes
    if init_k < 2 * c:
        errors.append(f"init-k={init_k} < 2*classes={2 * c} "
                      f"(need >= {2 * c} for frozen+stones).")
    if init_k > cfg.K_max:
        errors.append(f"init-k={init_k} > K_max={cfg.K_max}.")
    if init_d < 2:
        errors.append(f"init-d={init_d} < 2.")
    if init_d > cfg.D_max:
        errors.append(f"init-d={init_d} > D_max={cfg.D_max}.")
    if epochs < 1:
        errors.append(f"epochs={epochs} < 1.")
    return errors


# ---------------------------------------------------------------------------
# Live two-point time calibration
# ---------------------------------------------------------------------------
def _calibrate(make_cfg, n1: int = 5, n2: int = 15) -> Dict[str, float]:
    """Measure compile + per-epoch step time on this machine for this config.

    Runs training twice with growth disabled (n1 and n2 epochs).  The forced
    last-epoch diagnostic and the one-time JIT compile appear in BOTH runs, so
    the difference isolates the pure per-epoch step cost:

        per_epoch = (T2 - T1) / (n2 - n1)
        compile   = T1 - per_epoch * n1     (compile + one diagnostic tick)
    """
    from .train import train

    def _timed(n_epochs):
        cfg = make_cfg(n_epochs=n_epochs, min_epochs_before_grow=10 ** 9)
        t0 = time.time()
        train(cfg, verbose=False)
        return time.time() - t0

    t1 = _timed(n1)
    t2 = _timed(n2)
    per_epoch = max((t2 - t1) / max(n2 - n1, 1), 1e-6)
    compile_s = max(t1 - per_epoch * n1, 0.0)
    return {"per_epoch": per_epoch, "compile": compile_s, "t1": t1, "t2": t2}


def _estimate(calib: Dict[str, float], cfg, epochs: int) -> Dict[str, float]:
    """Turn calibration numbers into a (lower, upper) wall-time estimate."""
    per_epoch = calib["per_epoch"]
    compile_s = calib["compile"]
    lower = compile_s + per_epoch * epochs

    # Contingency for autonomous growth: each grow event recompiles (~one more
    # compile) and raises per-epoch cost as K/D climb.  We cap the number of
    # grows by the configured ceilings and apply a soft per-epoch inflation.
    n_grow_max = ((cfg.K_max - cfg.K_init) // max(cfg.K_grow, 1)
                  + (cfg.D_max - cfg.D_init))
    upper = (compile_s * (1 + n_grow_max)
             + per_epoch * epochs * 1.4)
    return {"lower": lower, "upper": upper, "n_grow_max": int(n_grow_max)}


def _fmt_secs(s: float) -> str:
    if s < 90:
        return f"{s:.0f}s"
    return f"{s/60:.1f}min"


# ---------------------------------------------------------------------------
# Evaluation summary
# ---------------------------------------------------------------------------
def _evaluate(state, cfg) -> Dict:
    from .data import load_dataset
    from .evaluate import classify, confusion_matrix

    spec = cfg.dataset_spec
    labels = list(cfg.class_labels)

    # canonical accuracy
    canon = {lab: spec.canonicals[lab] for lab in labels}
    canon_preds = {lab: classify(img, state, cfg)["pred"]
                   for lab, img in canon.items()}
    canon_correct = sum(1 for lab in labels if canon_preds[lab] == lab)

    # variant accuracy over the full training set (same seed)
    data = load_dataset(cfg.dataset, cfg.n_train_per_class, cfg.dataset_seed)
    imgs, labs = [], []
    for lab in labels:
        for im in data["images_by_label"][lab]:
            imgs.append(im)
            labs.append(lab)
    cm = confusion_matrix(imgs, labs, state, cfg)
    return {
        "labels": labels,
        "canon_preds": canon_preds,
        "canon_correct": canon_correct,
        "canon_total": len(labels),
        "variant_overall": cm["overall_acc"],
        "variant_per_class": dict(zip(cm["labels"], cm["per_class_acc"])),
        "confusion": cm["matrix"],
    }


def _write_results_md(path, exp_id, cfg, args, device_kind, eval_d,
                      final_D, final_K, elapsed, est, growth_log,
                      gallery_name, terrain_name, final_attractor_sigma=None):
    labels = eval_d["labels"]
    gk = sum(1 for e in growth_log if e["event"] == "grow_K")
    gd = sum(1 for e in growth_log if e["event"] == "grow_D")

    lines = []
    lines.append(f"# CHM 실험 결과 — exp_{exp_id:03d}\n")
    lines.append("## 실험 설정")
    lines.append(f"- 데이터셋: {args.dataset} ({cfg.dataset})")
    lines.append(f"- 지정 에폭: {args.epochs}")
    lines.append(f"- 초기 K: {args.init_k}")
    lines.append(f"- 초기 D: {args.init_d}")
    lines.append(f"- seed: {cfg.dataset_seed}")
    lines.append(f"- 끌개 레이아웃: {args.attractor} "
                 f"(σ_init={cfg.attractor_sigma_init}, frozen_w={cfg.frozen_w})")
    lines.append(f"- 끌개 σ 학습: {'on' if cfg.learn_attractor_sigma else 'off'}")
    lines.append(f"- 실행 디바이스: {device_kind}\n")

    lines.append("## 실험 결과")
    lines.append(f"- 분류 정확도 (canonical): "
                 f"{eval_d['canon_correct']}/{eval_d['canon_total']} "
                 f"({100*eval_d['canon_correct']/max(eval_d['canon_total'],1):.0f}%)")
    lines.append(f"- 분류 정확도 (variant 평균): "
                 f"{100*eval_d['variant_overall']:.1f}%")
    per_cls = ", ".join(f"{lab}={100*eval_d['variant_per_class'][lab]:.0f}%"
                        for lab in labels)
    lines.append(f"- 클래스별 정확도: {per_cls}")
    lines.append(f"- 최종 K: {final_K}")
    lines.append(f"- 최종 D: {final_D}")
    if final_attractor_sigma is not None:
        sig_str = ", ".join(f"{lab}={float(s):.2f}" for lab, s
                            in zip(labels, final_attractor_sigma))
        lines.append(f"- 최종 끌개 σ (학습 결과): {sig_str}")
    lines.append(f"- 소요 시간: {elapsed:.1f}초 "
                 f"(추정 {_fmt_secs(est['lower'])}~{_fmt_secs(est['upper'])})")
    lines.append(f"- 성장 이벤트: grow_K ×{gk}, grow_D ×{gd}\n")

    lines.append("## 혼동 행렬 (행=실제, 열=예측)")
    header = "|       | " + " | ".join(f"예측 {l}" for l in labels) + " |"
    sep = "|-------|" + "|".join("--------" for _ in labels) + "|"
    lines.append(header)
    lines.append(sep)
    cmx = eval_d["confusion"]
    for i, lab in enumerate(labels):
        row = " | ".join(str(int(cmx[i, j])) for j in range(len(labels)))
        lines.append(f"| 실제 {lab} | {row} |")
    lines.append("")

    lines.append("## 산출 파일")
    lines.append(f"- {gallery_name}")
    lines.append(f"- {terrain_name}")
    lines.append("- params.npz, config.json, history.json, growth_log.json")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(args) -> str:
    # kanzen submodules imported here (after device prescan at module load).
    from . import report_device
    from .config import Config
    from .train import train, save_run
    from .gallery import make_dataset_gallery
    from .terrain_evolution import make_terrain_evolution_3d

    dataset_key = _DATASET_ALIASES.get(args.dataset, args.dataset)

    # ---- device report ---------------------------------------------------
    device_kind = report_device()
    if _REQUESTED_DEVICE == "gpu" and device_kind != "GPU":
        print("[runner] 경고: --device gpu 요청했으나 GPU 미감지 -> CPU로 진행합니다.")

    # ---- build config ----------------------------------------------------
    preset = (_improved_attractor_preset(dataset_key)
              if args.attractor == "improved" else {})

    def make_cfg(**extra):
        kw = dict(K_init=args.init_k, D_init=args.init_d,
                  n_epochs=args.epochs, dataset_seed=args.seed,
                  learn_attractor_sigma=args.learn_sigma)
        kw.update(preset)        # improved attractor layout (if requested)
        kw.update(extra)
        return Config.with_dataset(dataset_key, **kw)

    cfg = make_cfg()

    # ---- validate --------------------------------------------------------
    errors = _validate(cfg, args.init_k, args.init_d, args.epochs)
    if errors:
        print("[runner] 입력 검증 실패:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(2)

    print(f"[runner] 설정: dataset={cfg.dataset} (classes={cfg.class_labels}), "
          f"epochs={args.epochs}, init_K={args.init_k}, init_D={args.init_d}, "
          f"seed={args.seed}")
    print(f"[runner] 끌개 레이아웃={args.attractor}, "
          f"σ학습={'on' if args.learn_sigma else 'off'} "
          f"(init={cfg.attractor_sigma_init}, frozen_w={cfg.frozen_w})")

    # ---- calibrate + estimate -------------------------------------------
    print("[runner] 캘리브레이션 측정 중 (성장 OFF, 짧은 2회 실행)...")
    calib = _calibrate(make_cfg)
    est = _estimate(calib, cfg, args.epochs)
    print(f"[runner] 컴파일~={calib['compile']:.1f}s, "
          f"정상구간~={calib['per_epoch']*1000:.1f} ms/epoch "
          f"(K={args.init_k}, D={args.init_d}, n_max={cfg.n_max})")
    print(f"[runner] 예상 소요: {_fmt_secs(est['lower'])} "
          f"(무성장) ~ {_fmt_secs(est['upper'])} "
          f"(성장 최대 {est['n_grow_max']}회 가정, 상한)")

    if args.dry_run:
        print("[runner] --dry-run: 추정만 하고 종료합니다.")
        return ""

    # ---- confirm ---------------------------------------------------------
    if not args.yes:
        try:
            ans = input("[runner] 실험을 진행할까요? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            print("[runner] 취소했습니다.")
            return ""

    # ---- experiment dir --------------------------------------------------
    out_dir, exp_id = _next_exp_dir(args.out_base)
    print(f"[runner] 실험 디렉토리: {out_dir}")

    # ---- 1) gallery ------------------------------------------------------
    gallery_path = os.path.join(out_dir, "dataset_gallery.png")
    make_dataset_gallery(cfg.dataset, n_per_class=cfg.n_train_per_class,
                         seed=cfg.dataset_seed, out_path=gallery_path)

    # ---- 2) train --------------------------------------------------------
    print("[runner] 학습 시작...")
    t0 = time.time()
    run_result = train(cfg, verbose=True)
    elapsed = time.time() - t0
    state = run_result["state"]
    history = run_result["history"]
    growth_log = run_result["growth_log"]
    print(f"[runner] 학습 완료: {elapsed:.1f}s "
          f"(추정 {_fmt_secs(est['lower'])}~{_fmt_secs(est['upper'])})")

    # ---- 3) terrain evolution -------------------------------------------
    terrain_path = os.path.join(out_dir, "terrain_evolution_3d.png")
    make_terrain_evolution_3d(history["terrain_snapshots"], cfg,
                              out_path=terrain_path)

    # ---- 4) evaluate -----------------------------------------------------
    print("[runner] 평가 중...")
    eval_d = _evaluate(state, cfg)

    # learned (or frozen) attractor sigma -- a white-box interpretability asset
    import numpy as np
    import jax.nn as jnn
    final_attractor_sigma = np.asarray(
        jnn.softplus(state.params["attractor_sigma_raw"]) + 0.1)

    # ---- 5) save ---------------------------------------------------------
    save_run(out_dir, run_result)
    _write_results_md(
        os.path.join(out_dir, "results.md"), exp_id, cfg, args, device_kind,
        eval_d, state.D, state.K_learn, elapsed, est, growth_log,
        "dataset_gallery.png", "terrain_evolution_3d.png",
        final_attractor_sigma)

    # ---- summary ---------------------------------------------------------
    print("\n=== 실험 완료 ===")
    print(f"정확도(canonical): {eval_d['canon_correct']}/{eval_d['canon_total']}"
          f"  |  정확도(variant): {100*eval_d['variant_overall']:.1f}%")
    print(f"최종 K: {state.K_learn}   최종 D: {state.D}")
    sig_str = ", ".join(f"{lab}={s:.2f}" for lab, s
                        in zip(eval_d["labels"], final_attractor_sigma))
    print(f"끌개 레이아웃={args.attractor}, σ학습={'on' if args.learn_sigma else 'off'} "
          f"-> 최종 끌개 σ: {sig_str}")
    print(f"추정: {_fmt_secs(est['lower'])}~{_fmt_secs(est['upper'])}"
          f"  ->  실측: {elapsed:.1f}s")
    gk = sum(1 for e in growth_log if e["event"] == "grow_K")
    gd = sum(1 for e in growth_log if e["event"] == "grow_D")
    print(f"성장: grow_K x{gk}, grow_D x{gd}")
    print(f"저장: {out_dir}/")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="kanzen.runner",
        description="CHM ready-to-experiment CLI runner")
    parser.add_argument("--dataset", default="OX",
                        help="OX | ABC | abcd (or OX_8/ABC_16/abcd_32)")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--init-k", dest="init_k", type=int, default=16)
    parser.add_argument("--init-d", dest="init_d", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"],
                        default="auto")
    parser.add_argument("--attractor", choices=["default", "improved"],
                        default="default",
                        help="attractor layout: dataset default, or the "
                             "data-center-symmetric 'improved' preset")
    parser.add_argument("--learn-sigma", dest="learn_sigma",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="learn the attractor sigma (--no-learn-sigma to "
                             "freeze it at init)")
    parser.add_argument("--out-base", dest="out_base", default=None,
                        help="experiment base dir (default research/experiment)")
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="only show the time estimate, then exit")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
