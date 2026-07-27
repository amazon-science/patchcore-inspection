"""継続学習9条件の fps 計測サマリーを可視化するスクリプト。

Usage:
    PYTHONPATH=./src python scripts/plot_cl_fps.py \
        --summary results/ContinualLearning/fps_<TS>/summary.csv \
        --out     results/ContinualLearning/fps_<TS>/fps_summary.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK JP"

# コアセット率ごとの色（1% / 5% / 10%）
COLORS = {1: "tab:green", 5: "tab:orange", 10: "tab:blue"}
MARKERS = {500: "o", 1000: "s", 2000: "^"}


def load_summary(path: Path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "label": r["condition"],
                        "n": int(r["n_per_round"]),
                        "p": int(r["coreset_pct"]),
                        "bank": int(r["memory_bank_size"]),
                        "auroc": float(r["instance_auroc"]),
                        "fps": float(r["fps"]),
                    }
                )
            except (ValueError, KeyError):
                continue
    return rows


def plot(rows, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (1) メモリバンクサイズ vs fps
    ax = axes[0]
    for p in sorted({r["p"] for r in rows}):
        sub = sorted([r for r in rows if r["p"] == p], key=lambda x: x["bank"])
        ax.plot(
            [r["bank"] for r in sub],
            [r["fps"] for r in sub],
            marker="o",
            color=COLORS.get(p, "gray"),
            label=f"コアセット率 {p}%",
        )
    ax.set_xscale("log")
    ax.set_xlabel("メモリバンクサイズ [ベクトル数]")
    ax.set_ylabel("推論速度 [fps]")
    ax.set_title("メモリバンクサイズ vs 推論速度")
    ax.grid(alpha=0.3)
    ax.legend()

    # (2) 条件ごとの fps 棒グラフ
    ax = axes[1]
    ordered = sorted(rows, key=lambda x: (x["p"], x["n"]))
    ax.bar(
        range(len(ordered)),
        [r["fps"] for r in ordered],
        color=[COLORS.get(r["p"], "gray") for r in ordered],
    )
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([r["label"] for r in ordered], rotation=45, ha="right")
    ax.set_ylabel("推論速度 [fps]")
    ax.set_title("条件別の推論速度")
    ax.grid(alpha=0.3, axis="y")

    # (3) 精度 vs 速度のトレードオフ
    ax = axes[2]
    for r in rows:
        ax.scatter(
            r["fps"],
            r["auroc"],
            color=COLORS.get(r["p"], "gray"),
            marker=MARKERS.get(r["n"], "o"),
            s=80,
        )
        ax.annotate(
            r["label"], (r["fps"], r["auroc"]), fontsize=8,
            xytext=(4, 4), textcoords="offset points",
        )
    ax.set_xlabel("推論速度 [fps]")
    ax.set_ylabel("Instance AUROC")
    ax.set_title("精度 vs 推論速度のトレードオフ")
    ax.grid(alpha=0.3)

    fig.suptitle("継続学習9条件の推論速度（最終ラウンドモデル・テスト1,388枚）")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] 保存: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = load_summary(args.summary)
    if not rows:
        raise SystemExit(f"有効な行がありません: {args.summary}")
    plot(rows, args.out)


if __name__ == "__main__":
    main()
