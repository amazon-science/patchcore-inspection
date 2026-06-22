"""ベンチマーク結果を可視化するスクリプト。

Usage:
    PYTHONPATH=./src python scripts/visualize_benchmark.py [--results_dir DIR] [--out DIR]

Output（規約準拠・上書きしない）:
    <out>/<YYYYMMDD_HHMMSS>/benchmark_plot.png
                          /benchmark_plot.html
                          /meta.json /log.txt
    out 既定: <results_dir>/viz
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
from patchcore.runlog import Tee, jst_timestamp, make_run_dir, write_meta


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    # P01 -> 0.01, P1 -> 0.1; 末尾の _N は run index として無視
    pattern = re.compile(r"Bench_N(\d+)_P(\d+)(?:_\d+)*$")

    for csv_path in sorted(results_dir.glob("Bench_*/results.csv")):
        m = pattern.match(csv_path.parent.name)
        if not m:
            continue
        n = int(m.group(1))
        p = float("0." + m.group(2))

        df = pd.read_csv(csv_path)
        mean_row = df[df["Row Names"] == "Mean"].iloc[0]

        def safe(col):
            val = mean_row.get(col, np.nan)
            try:
                return float(val)
            except (ValueError, TypeError):
                return np.nan

        rows.append({
            "N": n,
            "P": p,
            "instance_auroc": safe("instance_auroc"),
            "full_pixel_auroc": safe("full_pixel_auroc"),
            "anomaly_pixel_auroc": safe("anomaly_pixel_auroc"),
            "train_time": safe("train_time"),
            "fps": safe("fps"),
            "memory_mb": safe("memory_mb"),
        })

    df = pd.DataFrame(rows).sort_values(["P", "N"]).reset_index(drop=True)
    return df


def plot_matplotlib(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PatchCore Benchmark: Scaling vs. Accuracy / Cost", fontsize=14)

    colors = {0.01: "#2196F3", 0.1: "#FF5722"}
    markers = {0.01: "o", 0.1: "s"}

    for p, grp in df.groupby("P"):
        valid = grp.dropna(subset=["instance_auroc"])
        label = f"P={p}"
        c = colors[p]
        m = markers[p]

        # Panel 1: instance_auroc vs N
        axes[0, 0].plot(valid["N"], valid["instance_auroc"], marker=m, color=c, label=label)
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_xlabel("max_train_images")
        axes[0, 0].set_ylabel("instance_auroc")
        axes[0, 0].set_title("Accuracy (instance AUROC) vs N")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Panel 2: train_time vs N
        valid_t = grp.dropna(subset=["train_time"])
        axes[0, 1].plot(valid_t["N"], valid_t["train_time"], marker=m, color=c, label=label)

        # Panel 3: memory_mb vs N
        valid_m = grp.dropna(subset=["memory_mb"])
        axes[1, 0].plot(valid_m["N"], valid_m["memory_mb"], marker=m, color=c, label=label)

    # train_time axes
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("max_train_images")
    axes[0, 1].set_ylabel("train_time (sec)")
    axes[0, 1].set_title("Training Time vs N")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # memory axes
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("max_train_images")
    axes[1, 0].set_ylabel("memory_mb")
    axes[1, 0].set_title("Memory Usage vs N")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: tradeoff scatter (train_time vs instance_auroc)
    for p, grp in df.groupby("P"):
        valid = grp.dropna(subset=["train_time", "instance_auroc"])
        axes[1, 1].scatter(valid["train_time"], valid["instance_auroc"],
                           color=colors[p], marker=markers[p], label=f"P={p}", zorder=3)
        for _, row in valid.iterrows():
            axes[1, 1].annotate(
                f"N={int(row['N'])}",
                (row["train_time"], row["instance_auroc"]),
                fontsize=7, xytext=(4, 2), textcoords="offset points",
            )

    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("train_time (sec)")
    axes[1, 1].set_ylabel("instance_auroc")
    axes[1, 1].set_title("Tradeoff: Accuracy vs Training Time")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG saved: {out_path}")


def plot_plotly(df: pd.DataFrame, out_path: Path) -> None:
    colors = {0.01: "#2196F3", 0.1: "#FF5722"}
    symbols = {0.01: "circle", 0.1: "square"}

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Accuracy (instance AUROC) vs N",
            "Training Time vs N",
            "Memory Usage vs N",
            "Tradeoff: Accuracy vs Training Time",
        ],
    )

    for p, grp in df.groupby("P"):
        c = colors[p]
        sym = symbols[p]
        label = f"P={p}"

        # Panel 1
        valid = grp.dropna(subset=["instance_auroc"])
        fig.add_trace(go.Scatter(
            x=valid["N"], y=valid["instance_auroc"],
            mode="lines+markers", name=label, legendgroup=label,
            marker=dict(symbol=sym, size=8), line=dict(color=c),
        ), row=1, col=1)

        # Panel 2
        valid_t = grp.dropna(subset=["train_time"])
        fig.add_trace(go.Scatter(
            x=valid_t["N"], y=valid_t["train_time"],
            mode="lines+markers", name=label, legendgroup=label, showlegend=False,
            marker=dict(symbol=sym, size=8), line=dict(color=c),
        ), row=1, col=2)

        # Panel 3
        valid_m = grp.dropna(subset=["memory_mb"])
        fig.add_trace(go.Scatter(
            x=valid_m["N"], y=valid_m["memory_mb"],
            mode="lines+markers", name=label, legendgroup=label, showlegend=False,
            marker=dict(symbol=sym, size=8), line=dict(color=c),
        ), row=2, col=1)

        # Panel 4
        valid_tr = grp.dropna(subset=["train_time", "instance_auroc"])
        fig.add_trace(go.Scatter(
            x=valid_tr["train_time"], y=valid_tr["instance_auroc"],
            mode="markers+text",
            text=[f"N={int(n)}" for n in valid_tr["N"]],
            textposition="top right", textfont=dict(size=9),
            name=label, legendgroup=label, showlegend=False,
            marker=dict(symbol=sym, size=10, color=c),
        ), row=2, col=2)

    fig.update_xaxes(type="log", title_text="max_train_images", row=1, col=1)
    fig.update_yaxes(title_text="instance_auroc", row=1, col=1)

    fig.update_xaxes(type="log", title_text="max_train_images", row=1, col=2)
    fig.update_yaxes(type="log", title_text="train_time (sec)", row=1, col=2)

    fig.update_xaxes(type="log", title_text="max_train_images", row=2, col=1)
    fig.update_yaxes(type="log", title_text="memory_mb", row=2, col=1)

    fig.update_xaxes(type="log", title_text="train_time (sec)", row=2, col=2)
    fig.update_yaxes(title_text="instance_auroc", row=2, col=2)

    fig.update_layout(
        title="PatchCore Benchmark: Scaling vs. Accuracy / Cost",
        height=800,
        legend=dict(groupclick="toggleitem"),
    )

    fig.write_html(str(out_path))
    print(f"HTML saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/Benchmark_Results",
                        help="ベンチマーク結果（Bench_*/results.csv）のルートディレクトリ")
    parser.add_argument("--out", default=None,
                        help="出力先の親（既定: <results_dir>/viz）。実体は <out>/<timestamp>/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_base = args.out or str(results_dir / "viz")
    ts = jst_timestamp()
    run_dir = Path(make_run_dir(out_base, ts))

    with Tee(str(run_dir)):
        print(f"[results_dir] {results_dir}")
        print(f"[out]         {run_dir}")
        df = load_results(results_dir)
        print(f"Loaded {len(df)} rows:\n{df.to_string()}\n")

        plot_matplotlib(df, run_dir / "benchmark_plot.png")
        plot_plotly(df, run_dir / "benchmark_plot.html")
    write_meta(str(run_dir), {"results_dir": str(results_dir), "n_rows": int(len(df))})
    print(f"[meta] {run_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
