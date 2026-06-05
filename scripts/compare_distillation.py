#!/usr/bin/env python3
"""蒸留手法比較のプロット生成。

run_distillation.py が出力した results.csv（method, auroc, f1, memory_mb, fps, ...）を
読み、AUROC×メモリのトレードオフ散布図と各指標の棒グラフを描く。

usage: PYTHONPATH=./src python scripts/compare_distillation.py <run_dir>
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False


def load_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "method": r["method"],
                    "auroc": float(r["auroc"]),
                    "f1": float(r["f1"]),
                    "memory_mb": float(r["memory_mb"]),
                    "fps": float(r["fps"]),
                }
            )
    return rows


def main():
    if len(sys.argv) < 2:
        print("usage: compare_distillation.py <run_dir>")
        sys.exit(1)
    run_dir = sys.argv[1]
    rows = load_rows(os.path.join(run_dir, "results.csv"))
    methods = [r["method"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("蒸留手法の比較 — AUROC × メモリ × 速度")

    # (1) AUROC vs メモリ（対数軸）の散布図
    ax = axes[0]
    for r in rows:
        ax.scatter(r["memory_mb"], r["auroc"], s=80)
        ax.annotate(r["method"], (r["memory_mb"], r["auroc"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("メモリ (MB, 対数)")
    ax.set_ylabel("Instance AUROC")
    ax.set_title("AUROC vs メモリ")
    ax.grid(True, which="both", alpha=0.3)

    # (2) メモリ（対数軸の棒グラフ）
    ax = axes[1]
    ax.bar(methods, [r["memory_mb"] for r in rows], color="#1f77b4")
    ax.set_yscale("log")
    ax.set_ylabel("メモリ (MB, 対数)")
    ax.set_title("メモリフットプリント")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)

    # (3) AUROC の棒グラフ
    ax = axes[2]
    aurocs = [r["auroc"] for r in rows]
    ax.bar(methods, aurocs, color="#2ca02c")
    ax.set_ylim(min(aurocs) - 0.02, max(aurocs) + 0.01)
    ax.set_ylabel("Instance AUROC")
    ax.set_title("精度")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(run_dir, "comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


if __name__ == "__main__":
    main()
