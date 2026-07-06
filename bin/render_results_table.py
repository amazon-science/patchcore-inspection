"""蒸留の結果表をスライド体裁（緑ヘッダ・グループ見出し）の PNG で描画する。

精度は aggregate_distill_final.py の comparison.csv、性能は bench_networks.py の
networks.csv（pipeline=特徴生成+異常生成の合計）から読む。P5(③) は後段凍結(③A)/
後段再学習(③B)の2行に分けて示す。列: 条件/入力/学習データ | map_MSE/map_MAE |
Instance AUROC/AP/F1max | 推論時間(ms)/メモリ(MB)。
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import patchcore.runlog as runlog

plt.rcParams["font.family"] = "Noto Sans CJK JP"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GREEN = "#8CB63C"
LGREEN = "#C6E0A4"
PEACH = "#F8CBAD"
GRAY = "#F2F2F2"


def parse_args():
    p = argparse.ArgumentParser(description="蒸留 結果表の PNG 描画")
    p.add_argument("--compare_csv", required=True, help="aggregate_distill_final.py の comparison.csv")
    p.add_argument("--bench_csv", required=True, help="bench_networks.py の networks.csv")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map"))
    p.add_argument("--name", default="results_table")
    return p.parse_args()


def _load_accuracy(path):
    acc = {}
    for r in csv.DictReader(open(path)):
        acc[r["id"]] = r
    return acc


def _load_perf(path):
    lat = defaultdict(float)
    size = defaultdict(float)
    for r in csv.DictReader(open(path)):
        lat[r["condition"]] += float(r["latency_ms"] or 0)
        size[r["condition"]] += float(r["model_size_mb"] or 0)
    return lat, size


def _fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def main():
    args = parse_args()
    acc = _load_accuracy(args.compare_csv)
    lat, size = _load_perf(args.bench_csv)
    run_dir = runlog.make_run_dir(args.out, timestamp=f"{args.name}_{runlog.jst_timestamp()}")

    # 行定義: (条件, 入力, 学習, acc_id, perf_key, 強調)
    ROWS = [
        ("T0 (Teacher)", "PatchCore", "正常のみ", "T0", "T0", False),
        ("P1", "画像", "正常のみ", "P1", "P1", False),
        ("P2", "画像", "正常+疑似", "P2", "P2", False),
        ("P3", "特徴マップ", "正常のみ", "P3", "P3", False),
        ("P4", "特徴マップ", "正常+疑似", "P4", "P4", False),
        ("P5-A 画像→特徴\n(後段 凍結)", "画像", "正常+疑似", "P5A", "③", True),
        ("P5-B 画像→特徴\n(後段 再学習)", "画像", "正常+疑似", "P5B", "③", True),
    ]

    def cells(cond, inp, tr, aid, pkey):
        a = acc.get(aid, {})
        is_t0 = aid == "T0"
        mmse = "0（基準）" if is_t0 else _fmt(a.get("map_mse"), 6)
        mmae = "0（基準）" if is_t0 else _fmt(a.get("map_mae"), 6)
        return [cond, inp, tr, mmse, mmae,
                _fmt(a.get("auroc")), _fmt(a.get("ap")), _fmt(a.get("f1_max")),
                f"{lat[pkey]:.1f}", f"{size[pkey]:.1f}"]

    table = [cells(*r[:5]) for r in ROWS]
    highlight = [r[5] for r in ROWS]

    # 列幅（相対）
    widths = [2.0, 1.1, 1.4, 1.35, 1.35, 1.7, 0.85, 1.0, 1.3, 1.15]
    xs = [0]
    for w in widths:
        xs.append(xs[-1] + w)
    total_w = xs[-1]
    n_data = len(table)
    row_h = 1.0
    head_h = 1.0
    sub_h = 0.9
    total_h = head_h + sub_h + n_data * row_h

    fig, ax = plt.subplots(figsize=(total_w * 1.15, total_h * 0.62))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    ax.invert_yaxis()

    def cell(cx0, cx1, cy0, cy1, text, face, bold=False, fs=11, color="black"):
        ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0,
                               facecolor=face, edgecolor="white", lw=1.5))
        ax.text((cx0 + cx1) / 2, (cy0 + cy1) / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color=color)

    # --- グループ見出し（1段目） ---
    y0, y1 = 0, head_h
    cell(xs[0], xs[3], 0, head_h + sub_h, "", GREEN)  # 条件/入力/学習は2段ぶち抜き
    cell(xs[0], xs[1], 0, head_h + sub_h, "条件", GREEN, fs=12, color="white")
    cell(xs[1], xs[2], 0, head_h + sub_h, "入力", GREEN, fs=12, color="white")
    cell(xs[2], xs[3], 0, head_h + sub_h, "学習データ", GREEN, fs=12, color="white")
    cell(xs[3], xs[5], y0, y1, "異常マップに対する評価", GREEN, fs=12, color="white")
    cell(xs[5], xs[8], y0, y1, "異常スコアに対する評価", GREEN, fs=12, color="white")
    cell(xs[8], xs[10], y0, y1, "パフォーマンス", GREEN, fs=12, color="white")

    # --- サブ見出し（2段目） ---
    y0, y1 = head_h, head_h + sub_h
    subs = ["map_MSE ↓", "map_MAE ↓", "Instance AUROC ↑", "AP ↑", "F1max ↑",
            "推論時間(ms)", "メモリ(MB)"]
    for k, t in enumerate(subs):
        cell(xs[3 + k], xs[4 + k], y0, y1, t, LGREEN, fs=10)

    # --- データ行 ---
    for i, (rowvals, hl) in enumerate(zip(table, highlight)):
        y0 = head_h + sub_h + i * row_h
        y1 = y0 + row_h
        base = PEACH if hl else (GRAY if i % 2 == 0 else "white")
        for c, val in enumerate(rowvals):
            fs = 9 if (c == 0 and "\n" in val) else 11
            cell(xs[c], xs[c + 1], y0, y1, val, base, bold=hl and c >= 3, fs=fs)

    fig.tight_layout(pad=0.4)
    out_png = os.path.join(run_dir, "results_table.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV も併せて出力
    cols = ["条件", "入力", "学習データ", "map_MSE", "map_MAE",
            "Instance_AUROC", "AP", "F1max", "推論時間_ms", "メモリ_MB"]
    with open(os.path.join(run_dir, "results_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in table:
            w.writerow([c.replace("\n", " ") for c in row])

    print(f"保存: {out_png}")
    runlog.write_meta(run_dir, extra={"compare_csv": args.compare_csv, "bench_csv": args.bench_csv})
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
