#!/usr/bin/env python3
"""疑似異常が「教師から見てどれくらい異常か」を実データと突き合わせる。

Teacher(PatchCore) スコアの分布を
  正常 test / 実異常 test / 各疑似バリアント
で重ねて描く。疑似の難易度が実異常と揃っているほど、Student は実異常へ転移しやすい
という仮説を、学習結果と独立に検証するための図。
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import patchcore.runlog as runlog

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"


def parse_args():
    p = argparse.ArgumentParser(description="疑似異常と実異常の Teacher スコア分布比較")
    p.add_argument("--scores_csv", required=True,
                   help="eval の predictions/scores.csv（teacher_score と label を読む）")
    p.add_argument("--pack", nargs="+", required=True,
                   help="`ラベル=疑似B pack` の並び（scores.npy を読む）")
    p.add_argument("--name", default="score_alignment")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map", "score_alignment"))
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = runlog.make_run_dir(args.out)
    runlog.start_tee(run_dir)

    rows = np.genfromtxt(args.scores_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    label = np.array([int(r) for r in rows["label"]])
    tscore = np.array([float(r) for r in rows["teacher_score"]])
    series = [("実 test 正常", tscore[label == 0], "#4C72B0"),
              ("実 test 異常", tscore[label == 1], "#C44E52")]
    colors = ["#55A868", "#8172B2", "#CCB974", "#64B5CD"]
    for i, kv in enumerate(args.pack):
        lab, _, path = kv.partition("=")
        s = np.load(os.path.join(path, "scores.npy"))
        d = np.load(os.path.join(path, "done.npy"))
        s = s[d == 1]
        series.append((lab, s, colors[i % len(colors)]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.linspace(0, 20, 80)
    for lab, v, c in series:
        axes[0].hist(v, bins=bins, density=True, histtype="step", lw=2, label=lab, color=c)
        sv = np.sort(v)
        axes[1].plot(sv, np.linspace(0, 1, len(sv)), lw=2, label=lab, color=c)
    for ax, yl in zip(axes, ["確率密度", "累積割合"]):
        ax.set_xlabel("Teacher(PatchCore) 異常スコア")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("疑似異常の「難易度」は実異常と揃っているか（Teacher スコア分布）", fontsize=12)
    fig.tight_layout()
    f1 = os.path.join(run_dir, "score_alignment.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    csv_path = os.path.join(run_dir, "score_alignment.csv")
    with open(csv_path, "w") as f:
        f.write("series,n,mean,std,p10,median,p90\n")
        for lab, v, _ in series:
            f.write(f"{lab},{len(v)},{v.mean():.4f},{v.std():.4f},"
                    f"{np.percentile(v, 10):.4f},{np.median(v):.4f},{np.percentile(v, 90):.4f}\n")
            print(f"{lab:20s} n={len(v):7d} mean={v.mean():7.3f} median={np.median(v):7.3f} "
                  f"p10={np.percentile(v, 10):7.3f} p90={np.percentile(v, 90):7.3f}")
    print(f"図: {f1}\n表: {csv_path}")
    runlog.write_meta(run_dir, extra={"scores_csv": args.scores_csv, "pack": args.pack})


if __name__ == "__main__":
    main()
