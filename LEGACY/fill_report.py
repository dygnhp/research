"""
Fill the placeholders in kanzen/REPORT_FILL.md with actual experiment
results from experiments_out/all_summaries.json (or per-dataset files).

Run after run_experiments.py finishes.
"""
import json
import os
import sys

OUT_ROOT = "experiments_out_phaseA"
REPORT = "kanzen/REPORT_FILL.md"


def load_summary(name):
    path = os.path.join(OUT_ROOT, name, "experiment_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_canon(canonical_results, labels):
    parts = []
    for lab in labels:
        r = canonical_results[lab]
        ok = "PASS" if r["pred"] == lab else f"FAIL→{r['pred']}"
        parts.append(f"{lab}→{r['pred']} ({ok})")
    return ", ".join(parts)


def fmt_per_class(per_class):
    return ", ".join(f"{lab}={int(round(acc*100))}%" for lab, acc in per_class.items())


def fmt_events(events):
    K_grows = sum(1 for e in events if e["event"] == "grow_K")
    D_grows = sum(1 for e in events if e["event"] == "grow_D")
    return f"K×{K_grows} + D×{D_grows}"


def main():
    summaries = {}
    for name in ("OX_8", "ABC_16", "abcd_32"):
        s = load_summary(name)
        if s is None:
            print(f"WARNING: no summary for {name}, skipping")
            continue
        summaries[name] = s

    with open(REPORT, "r", encoding="utf-8") as f:
        text = f.read()

    # ---- (2) Training table ------------------------------------------------
    label_map = {"OX_8": "OX", "ABC_16": "ABC", "abcd_32": "abcd"}
    for ds, prefix in label_map.items():
        s = summaries.get(ds)
        if s is None:
            continue
        text = text.replace(f"{{{{{prefix}_train_time}}}}", f"{s['train_time_s']:.0f}s")
        text = text.replace(f"{{{{{prefix}_D}}}}",          str(s["final_D"]))
        text = text.replace(f"{{{{{prefix}_K}}}}",          str(s["final_K_learn"]))
        text = text.replace(f"{{{{{prefix}_loss}}}}",       f"{s['final_loss']:.1f}")
        text = text.replace(f"{{{{{prefix}_events}}}}",     fmt_events(s["growth_events"]))

    # ---- (3) Accuracy table -----------------------------------------------
    for ds, prefix in label_map.items():
        s = summaries.get(ds)
        if s is None:
            continue
        labels = s["variant_accuracy"]["labels"]
        text = text.replace(f"{{{{{prefix}_canon}}}}",   fmt_canon(s["canonical_results"], labels))
        text = text.replace(f"{{{{{prefix}_var_acc}}}}", f"{int(round(s['variant_accuracy']['overall']*100))}%")
        text = text.replace(f"{{{{{prefix}_var_per}}}}", fmt_per_class(s["variant_accuracy"]["per_class"]))

    # ---- (4) Append a detailed sweep tables block at the end --------------
    block_lines = ["\n\n## 4-가. 부록: 상세 실측 표\n"]
    for ds, prefix in label_map.items():
        s = summaries.get(ds)
        if s is None:
            continue
        block_lines.append(f"\n### {ds}\n")
        block_lines.append(f"- **학습 시간**: {s['train_time_s']:.1f}s "
                           f"({s['n_epochs']} epoch, "
                           f"{1000*s['train_time_s']/s['n_epochs']:.0f} ms/epoch)\n")
        block_lines.append(f"- **최종 (D, K_learn)**: ({s['final_D']}, {s['final_K_learn']})\n")
        block_lines.append(f"- **성장 이벤트 시퀀스**:\n")
        for ev in s["growth_events"]:
            if ev["event"] == "grow_K":
                block_lines.append(f"  - ep {ev['epoch']}: grow_K → K_learn = {ev['K_learn_after']}\n")
            else:
                block_lines.append(f"  - ep {ev['epoch']}: grow_D → D = {ev['D_after']}\n")
        if not s["growth_events"]:
            block_lines.append("  - (no growth events)\n")
        block_lines.append(f"- **최종 진단량**: ")
        fd = s.get("final_diagnostics", {})
        block_lines.append(
            f"eps_q_max={fd.get('eps_q_max', '?'):.2f}, "
            f"eps_p_max={fd.get('eps_p_max', '?'):.2f}, "
            f"R2_min={fd.get('R2_min', '?'):.2f}\n"
        )
        block_lines.append(f"- **Variant accuracy 혼동 행렬** (행=정답, 열=예측):\n\n")
        labels = s["variant_accuracy"]["labels"]
        block_lines.append("| 정답＼예측 | " + " | ".join(labels) + " |\n")
        block_lines.append("|" + "---|" * (len(labels) + 1) + "\n")
        for i, lab in enumerate(labels):
            row = " | ".join(str(s["variant_accuracy"]["matrix"][i][j]) for j in range(len(labels)))
            block_lines.append(f"| **{lab}** | {row} |\n")
        block_lines.append("\n- **Gamma sweep**: ")
        gg = s["gamma_sweep"]
        block_lines.append(", ".join(f"γ={g}→{int(round(a*100))}%"
                                      for g, a in zip(gg["gammas"], gg["acc"])) + "\n")
        block_lines.append("- **Ablation**: ")
        block_lines.append(", ".join(f"{k}={int(round(v['acc']*100))}%"
                                      for k, v in s["ablation"].items()) + "\n")
        block_lines.append("- **Shift sweep** (max±2): ")
        block_lines.append(", ".join(f"{lab}={int(round(d['acc']*100))}%"
                                      for lab, d in s["shift_sweep"].items()) + "\n")
        block_lines.append("- **Noise sweep** (캐노니컬에 픽셀 flip): ")
        for lab, d in s["noise_sweep"].items():
            block_lines.append(f"\n  - {lab}: " + " → ".join(
                f"L{lvl}:{int(round(a*100))}%" for lvl, a in zip(d["levels"], d["acc"])))
        block_lines.append("\n\n")

    text += "".join(block_lines)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"OK: filled {REPORT} with {len(summaries)} dataset(s)")


if __name__ == "__main__":
    main()
