#!/usr/bin/env python3
"""疑似異常の「形 × 境界」2×2 要因計画を、スケール別にまとめて描く。

各セルは複数シードの平均±標準偏差。効果量（境界ぼかしの主効果／形の主効果）を
数値で併記し、どちらが精度を動かしたのかを 1 枚で示す。
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
SHAPES = ["poly", "irregular"]
EDGES = ["hard", "soft"]


def parse_args():
    p = argparse.ArgumentParser(description="形×境界 2×2 要因計画の図")
    p.add_argument("--cell", nargs="+", required=True,
                   help="`スケール:形:境界=eval1,eval2,...` の並び（例 10x:poly:hard=results/...）")
    p.add_argument("--teacher", type=float, default=0.9282, help="参照線（Teacher AUROC）")
    p.add_argument("--name", default="shape_factorial")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map", "shape_factorial"))
    return p.parse_args()


def _auroc(eval_dir):
    with open(os.path.join(eval_dir, "metrics", "image_metrics.json")) as f:
        return json.load(f)["student"]["auroc"]


def main():
    args = parse_args()
    run_dir = runlog.make_run_dir(args.out)
    runlog.start_tee(run_dir)

    cells = {}
    for item in args.cell:
        key, _, dirs = item.partition("=")
        scale, shape, edge = key.split(":")
        vals = [_auroc(d) for d in dirs.split(",") if d]
        cells[(scale, shape, edge)] = np.array(vals)

    scales = sorted({k[0] for k in cells}, key=lambda s: int(s.rstrip("x")))
    fig, axes = plt.subplots(1, len(scales), figsize=(6.0 * len(scales), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(SHAPES))
    w = 0.34
    colors = {"hard": "#A0A0A0", "soft": "#4C72B0"}
    rows = []
    for ax, scale in zip(axes, scales):
        for i, edge in enumerate(EDGES):
            m = [cells[(scale, s, edge)].mean() if (scale, s, edge) in cells else np.nan for s in SHAPES]
            e = [cells[(scale, s, edge)].std(ddof=1) if (scale, s, edge) in cells
                 and len(cells[(scale, s, edge)]) > 1 else 0.0 for s in SHAPES]
            ax.bar(x + (i - 0.5) * w, m, w, yerr=e, capsize=4, label=f"境界 {edge}",
                   color=colors[edge], error_kw={"ecolor": "#333", "lw": 1.1})
            for xx, mm, ee, s in zip(x + (i - 0.5) * w, m, e, SHAPES):
                n = len(cells.get((scale, s, edge), []))
                ax.text(xx, mm + ee + 0.012, f"{mm:.3f}\n±{ee:.3f} (n={n})", ha="center", fontsize=8)
                rows.append({"scale": scale, "shape": s, "edge": edge, "n": n,
                             "auroc_mean": mm, "auroc_std": ee})
        ax.axhline(args.teacher, ls="--", color="crimson", lw=1.3,
                   label=f"Teacher = {args.teacher:.4f}")
        ax.set_xticks(x)
        ax.set_xticklabels(["poly（従来の凸多角形）", "irregular（実欠陥に整合）"], fontsize=9)
        ax.set_xlabel("疑似異常の形")
        ax.grid(axis="y", alpha=0.3)
        # 主効果はセル平均どうしの差の平均（シード数が不均衡でも偏らない）
        def cell(sh, ed):
            v = cells.get((scale, sh, ed))
            return float(v.mean()) if v is not None and len(v) else np.nan

        d_edge = float(np.nanmean([cell(sh, "soft") - cell(sh, "hard") for sh in SHAPES]))
        d_shape = float(np.nanmean([cell("irregular", ed) - cell("poly", ed) for ed in EDGES]))
        ax.set_title(f"{scale}（{'23,888' if scale == '1x' else '238,880'}枚）\n"
                     f"主効果：境界ぼかし {d_edge:+.3f} / 形 {d_shape:+.3f}", fontsize=11)
    axes[0].set_ylabel("Instance AUROC")
    axes[0].set_ylim(0.7, 1.02)
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("疑似異常の「形」と「境界」どちらが精度を動かすか（① image→map・B方式教師）", fontsize=13)
    fig.tight_layout()
    f1 = os.path.join(run_dir, "shape_factorial.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    csv_path = os.path.join(run_dir, "factorial.csv")
    with open(csv_path, "w") as f:
        f.write("scale,shape,edge,n_seeds,auroc_mean,auroc_std\n")
        for r in rows:
            f.write(f"{r['scale']},{r['shape']},{r['edge']},{r['n']},"
                    f"{r['auroc_mean']:.4f},{r['auroc_std']:.4f}\n")
            print(f"{r['scale']:4s} {r['shape']:10s} {r['edge']:5s} n={r['n']} "
                  f"AUROC={r['auroc_mean']:.4f}±{r['auroc_std']:.4f}")
    print(f"図: {f1}\n表: {csv_path}")
    runlog.write_meta(run_dir, extra={"cells": {":".join(k): list(map(float, v)) for k, v in cells.items()},
                                      "teacher": args.teacher})


if __name__ == "__main__":
    main()
