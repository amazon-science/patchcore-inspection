"""ContinualLearning 実験群を比較するプロット生成スクリプト.

results/ContinualLearning/<timestamp>/results.csv を読み込み、
コアセット率 P × ラウンド数 R の 3x3 グリッドとして比較図を出力する。
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "ContinualLearning",
)

# (timestamp): (percentage, n_rounds)。bank_size の増分から確定した対応表。
RUNS = {
    "20260526_203259": (1.0, 11),
    "20260526_233821": (1.0, 22),
    "20260527_020716": (1.0, 44),
    "20260527_050630": (0.5, 11),
    "20260527_064537": (0.5, 22),
    "20260527_080922": (0.5, 44),
    "20260527_095500": (0.1, 11),
    "20260527_102726": (0.1, 22),
    "20260527_110125": (0.1, 44),
}

P_COLOR = {1.0: "#1f77b4", 0.5: "#ff7f0e", 0.1: "#2ca02c"}
R_STYLE = {11: "-", 22: "--", 44: ":"}
R_MARKER = {11: "o", 22: "s", 44: "^"}


def load(ts):
    rows = []
    with open(os.path.join(BASE, ts, "results.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "round": int(r["round"]),
                    "instance_auroc": float(r["instance_auroc"]),
                    "train_time": float(r["train_time"]),
                    "memory_mb": float(r["memory_mb"]),
                    "memory_bank_size": int(r["memory_bank_size"]),
                }
            )
    return rows


def label(p, r):
    return f"P={p}, R={r}"


def main():
    data = {ts: load(ts) for ts in RUNS}

    # ---- 図1: 4指標を進捗(データ割合)軸で重ねる ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Continual Learning 比較 — コアセット率 P × ラウンド数 R (総データ量一定)",
        fontsize=14,
    )
    metrics = [
        ("instance_auroc", "Instance AUROC", "AUROC"),
        ("train_time", "Train Time per Round", "Seconds"),
        ("memory_mb", "Process Memory", "MB"),
        ("memory_bank_size", "Memory Bank Size", "# Vectors"),
    ]
    for ax, (key, title, ylab) in zip(axes.flat, metrics):
        for ts, (p, r) in RUNS.items():
            rows = data[ts]
            frac = [row["round"] / r for row in rows]
            ax.plot(
                frac,
                [row[key] for row in rows],
                R_STYLE[r],
                color=P_COLOR[p],
                marker=R_MARKER[r],
                markersize=3,
                linewidth=1.3,
                label=label(p, r),
            )
        ax.set_title(title)
        ax.set_xlabel("学習データ投入割合 (round / n_rounds)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=3, loc="lower right")
    plt.tight_layout()
    out1 = os.path.join(BASE, "comparison_curves.png")
    plt.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"[saved] {out1}")

    # ---- 図2: サマリ (最終AUROCヒートマップ / 精度-メモリ / 総学習時間) ----
    Ps = [1.0, 0.5, 0.1]
    Rs = [11, 22, 44]
    ts_of = {(p, r): ts for ts, (p, r) in RUNS.items()}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) 最終AUROCヒートマップ
    ax = axes[0]
    grid = [
        [data[ts_of[(p, r)]][-1]["instance_auroc"] for r in Rs] for p in Ps
    ]
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(Rs)))
    ax.set_xticklabels([f"R={r}" for r in Rs])
    ax.set_yticks(range(len(Ps)))
    ax.set_yticklabels([f"P={p}" for p in Ps])
    ax.set_title("最終ラウンド Instance AUROC")
    for i in range(len(Ps)):
        for j in range(len(Rs)):
            ax.text(
                j, i, f"{grid[i][j]:.4f}",
                ha="center", va="center", color="w", fontsize=10,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (b) 精度 vs メモリバンクサイズ (トレードオフ)
    ax = axes[1]
    for ts, (p, r) in RUNS.items():
        last = data[ts][-1]
        ax.scatter(
            last["memory_bank_size"],
            last["instance_auroc"],
            color=P_COLOR[p],
            marker=R_MARKER[r],
            s=90,
        )
        ax.annotate(
            label(p, r),
            (last["memory_bank_size"], last["instance_auroc"]),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_title("最終 AUROC vs メモリバンクサイズ (精度-メモリ)")
    ax.set_xlabel("Memory Bank Size (# Vectors, log)")
    ax.set_ylabel("Final Instance AUROC")
    ax.grid(True, alpha=0.3)

    # (c) 総学習時間 (グループ棒グラフ)
    ax = axes[2]
    width = 0.25
    x = range(len(Rs))
    for k, p in enumerate(Ps):
        totals = [sum(row["train_time"] for row in data[ts_of[(p, r)]]) for r in Rs]
        ax.bar(
            [xi + (k - 1) * width for xi in x],
            totals,
            width,
            color=P_COLOR[p],
            label=f"P={p}",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"R={r}" for r in Rs])
    ax.set_title("総学習時間 (全ラウンド合計)")
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out2 = os.path.join(BASE, "comparison_summary.png")
    plt.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"[saved] {out2}")


if __name__ == "__main__":
    main()
