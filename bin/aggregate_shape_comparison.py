#!/usr/bin/env python3
"""疑似異常「形状」バリアントの横並び比較表と図を作る。

入力＝`run_shape_screen_1x.sh` / 10x 本番の eval 出力フォルダ群。各 eval の
image_metrics / distillation_fidelity と、対応する疑似 pack の PatchCore
スコア統計（教師が疑似異常をどれだけ異常と見たか）を 1 表に集約する。
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import patchcore.runlog as runlog

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"


def parse_args():
    p = argparse.ArgumentParser(description="形状バリアントの比較表・図")
    p.add_argument("--eval", nargs="+", required=True,
                   help="`ラベル=eval フォルダ` の並び（例 S0_1x=results/.../eval_S0_1x_2026...）")
    p.add_argument("--pack", nargs="+", default=[],
                   help="`ラベル=疑似B pack` の並び（packed_meta.json の教師スコア統計を併記）")
    p.add_argument("--ref", nargs="+", default=["T0_Teacher=0.9282", "P2_inpainting926=0.8879"],
                   help="`ラベル=AUROC` の参照線（既存実験の確定値）")
    p.add_argument("--title", default="疑似異常の形状バリアント比較（① image→map・B方式教師）")
    p.add_argument("--name", default="shape_comparison")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map", "shape_comparison"))
    return p.parse_args()


def _kv(items):
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def _load(eval_dir):
    m = os.path.join(eval_dir, "metrics")
    with open(os.path.join(m, "image_metrics.json")) as f:
        im = json.load(f)
    row = {"auroc": im["student"]["auroc"], "ap": im["student"]["average_precision"],
           "f1max": im["student"]["f1_max"], "teacher_auroc": im["teacher"]["auroc"]}
    fid = os.path.join(m, "distillation_fidelity.json")
    if os.path.exists(fid):
        with open(fid) as f:
            d = json.load(f)
        row.update({"map_mse": d["map_mse"], "score_spearman": d["score_spearman"]})
    eff = os.path.join(m, "efficiency_metrics.json")
    if os.path.exists(eff):
        with open(eff) as f:
            d = json.load(f)["student"]
        row.update({"fps": d["fps"], "model_size_mb": d["model_size_mb"]})
    return row


def main():
    args = parse_args()
    run_dir = runlog.make_run_dir(args.out)
    runlog.start_tee(run_dir)

    evals, packs, refs = _kv(args.eval), _kv(args.pack), _kv(args.ref)
    rows = []
    for label, d in evals.items():
        r = {"label": label, **_load(d), "eval_dir": d}
        pk = packs.get(label)
        if pk:
            with open(os.path.join(pk, "packed_meta.json")) as f:
                pm = json.load(f)
            r.update({"pseudo_score_mean": pm.get("score_mean"), "pseudo_score_max": pm.get("score_max"),
                      "pseudo_count": pm.get("count")})
        rows.append(r)

    cols = ["label", "auroc", "ap", "f1max", "map_mse", "score_spearman",
            "pseudo_score_mean", "pseudo_score_max", "pseudo_count", "fps", "model_size_mb", "eval_dir"]
    csv_path = os.path.join(run_dir, "comparison.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"比較表: {csv_path}")
    print(f"{'条件':22s}{'AUROC':>9s}{'AP':>8s}{'F1max':>8s}{'map_MSE':>11s}{'疑似score平均':>14s}")
    for r in rows:
        print(f"{r['label']:22s}{r['auroc']:9.4f}{r['ap']:8.4f}{r['f1max']:8.4f}"
              f"{r.get('map_mse', float('nan')):11.2e}"
              f"{(r.get('pseudo_score_mean') or float('nan')):14.3f}")

    # --- 図：AUROC の横並び（参照線つき）---
    fig, ax = plt.subplots(figsize=(1.6 * len(rows) + 4, 4.6))
    labels = [r["label"] for r in rows]
    vals = [r["auroc"] for r in rows]
    bars = ax.bar(labels, vals, color="#4C72B0", width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}", ha="center", fontsize=10)
    styles = ["--", ":", "-."]
    for i, (rl, rv) in enumerate(refs.items()):
        ax.axhline(float(rv), ls=styles[i % len(styles)], color="crimson", lw=1.4,
                   label=f"{rl} = {float(rv):.4f}")
    ax.set_ylabel("Instance AUROC")
    ax.set_xlabel("疑似異常の形状バリアント")
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(args.title, fontsize=12)
    fig.tight_layout()
    f1 = os.path.join(run_dir, "auroc_by_shape.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    print(f"図: {f1}")

    runlog.write_meta(run_dir, extra={"eval": evals, "pack": packs, "ref": refs, "rows": rows})
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
