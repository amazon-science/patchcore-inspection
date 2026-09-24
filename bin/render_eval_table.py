#!/usr/bin/env python3
"""定量評価表（入出力 × 疑似異常データ）をスライド体裁の PNG で描く。

議事録スライド「評価結果③」と同じ列構成：
  入出力 / 疑似異常データ | 特徴mapに対する評価(map_MSE, map_MAE) |
  異常mapに対する評価(map_MSE, map_MAE) | 異常スコアに対する評価(AUROC, AP, F1max) |
  パフォーマンス(推論時間ms, メモリMB)
該当しないセルはグレーで塗る（入力 CSV で "-" と書く）。"未実施" はそのまま文字で残す。
"""
import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import patchcore.runlog as runlog

plt.rcParams["font.family"] = "Noto Sans CJK JP"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORANGE = "#ED9B33"
PEACH = "#FBE2CE"
GRAY = "#8C8C8C"
HILIGHT = "#DCE9F7"

GROUPS = [("", 2), ("特徴mapに対する評価", 2), ("異常mapに対する評価", 2),
          ("異常スコアに対する評価", 3), ("パフォーマンス", 2)]
SUBS = ["入出力", "疑似異常データ", "map_MSE ↓", "map_MAE ↓", "map_MSE ↓", "map_MAE ↓",
        "Instance AUROC ↑", "AP ↑", "F1max ↑", "推論時間(ms)", "メモリ使用量(MB)"]
WIDTHS = [1.35, 1.75, 1.05, 1.05, 1.05, 1.05, 1.25, 0.72, 0.78, 0.92, 1.05]


def parse_args():
    p = argparse.ArgumentParser(description="定量評価表の PNG 描画")
    p.add_argument("--csv", required=True,
                   help="列: io,pseudo,feat_mse,feat_mae,map_mse,map_mae,auroc,ap,f1max,ms,mb,"
                        "highlight(任意 1/0),bold(任意 カンマ区切り列名)")
    p.add_argument("--title", default="評価結果：疑似異常の生成方法による異常検知精度の違い")
    p.add_argument("--subtitle", default="")
    p.add_argument("--note", default="")
    p.add_argument("--name", default="eval_table")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map", "eval_table"))
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = runlog.make_run_dir(args.out)
    runlog.start_tee(run_dir)
    rows = list(csv.DictReader(open(args.csv)))
    keys = ["io", "pseudo", "feat_mse", "feat_mae", "map_mse", "map_mae",
            "auroc", "ap", "f1max", "ms", "mb"]

    n = len(rows)
    total_w = sum(WIDTHS)
    rh = 0.62
    head_h = 0.52
    fig_h = 1.5 + head_h * 2 + rh * n + (0.6 if args.note else 0)
    fig, ax = plt.subplots(figsize=(total_w * 1.55, fig_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    y_top = fig_h - 1.0
    # --- グループ見出し ---
    x = 0.0
    gi = 0
    for name, span in GROUPS:
        w = sum(WIDTHS[gi:gi + span])
        if name:
            ax.add_patch(Rectangle((x, y_top - head_h), w, head_h, fc=ORANGE, ec="white", lw=1.2))
            ax.text(x + w / 2, y_top - head_h / 2, name, ha="center", va="center",
                    fontsize=11, fontweight="bold")
        else:
            ax.add_patch(Rectangle((x, y_top - head_h * 2), w, head_h * 2, fc=ORANGE, ec="white", lw=1.2))
        x += w
        gi += span
    # --- 小見出し ---
    x = 0.0
    for i, (sub, w) in enumerate(zip(SUBS, WIDTHS)):
        y = y_top - head_h * 2
        if i < 2:
            ax.text(x + w / 2, y + head_h, sub, ha="center", va="center", fontsize=10.5, fontweight="bold")
        else:
            ax.add_patch(Rectangle((x, y), w, head_h, fc=ORANGE, ec="white", lw=1.2))
            ax.text(x + w / 2, y + head_h / 2, sub, ha="center", va="center", fontsize=9.5,
                    fontweight="bold")
        x += w

    # --- データ行 ---
    y = y_top - head_h * 2
    for r in rows:
        y -= rh
        x = 0.0
        base = PEACH if r.get("highlight") == "2" else (HILIGHT if r.get("highlight") == "1" else "white")
        bolds = set((r.get("bold") or "").split("|"))
        for k, w in zip(keys, WIDTHS):
            v = (r.get(k) or "").strip()
            fc = GRAY if v == "-" else base
            ax.add_patch(Rectangle((x, y), w, rh, fc=fc, ec="#9A9A9A", lw=0.8))
            if v and v != "-":
                ax.text(x + w / 2, y + rh / 2, v.replace("\\n", "\n"), ha="center", va="center",
                        fontsize=9 if k in ("io", "pseudo") else 9.5,
                        fontweight="bold" if k in bolds else "normal")
            x += w

    ax.text(0, fig_h - 0.42, args.title, fontsize=16, fontweight="bold", va="center")
    if args.subtitle:
        ax.text(0, fig_h - 0.82, args.subtitle, fontsize=11.5, va="center")
    if args.note:
        ax.text(0, y - 0.42, args.note.replace("\\n", "\n"), fontsize=9, va="top", color="#333333")

    fig.tight_layout()
    f1 = os.path.join(run_dir, f"{args.name}.png")
    fig.savefig(f1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"表: {f1}")
    runlog.write_meta(run_dir, extra={"csv": args.csv, "n_rows": n, "title": args.title})


if __name__ == "__main__":
    main()
