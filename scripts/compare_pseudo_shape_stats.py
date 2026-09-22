#!/usr/bin/env python3
"""疑似異常の「形」を実欠陥と定量比較する（形状バリアント S0/S2/S3 の設計検証）。

実欠陥マスク（参照）と疑似異常生成器のマスクから、欠陥ごとに
  面積率[%] / 外接矩形充填率 / 円形度 4πA/P² / 縦横比
を測って分布を比較する。形の改良が狙いどおり実欠陥へ寄ったかを、学習を回す前に判定する。

参照の扱い（リーク方針）：
  - フィット対象  = inpainting train GT（学習側に使ってよい）
  - 報告用のみ    = anomaly_dataset test GT（評価データなのでパラメータ調整には使わない）
"""
import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from scipy import ndimage

import patchcore.runlog as runlog
from patchcore.distill_map import pseudo_anomaly as PA

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"

METRICS = [("area_pct", "面積率 [%]"), ("extent", "外接矩形充填率"),
           ("circularity", "円形度 4πA/P²"), ("aspect", "縦横比")]

VARIANTS = {
    "S0_poly": {},
    "S2_irregular": {"shape_family": "irregular"},
    "S3_irregular_soft": {"shape_family": "irregular", "edge_blend": "soft"},
}


def parse_args():
    p = argparse.ArgumentParser(description="疑似異常の形状統計を実欠陥と比較")
    p.add_argument("--real_gt", default=os.path.join(REPO_ROOT, "dataset", "anomaly_dataset",
                                                     "ground_truth", "abnormal"),
                   help="実欠陥マスク（報告用参照・パラメータ調整には使わない）")
    p.add_argument("--fit_gt", default=os.path.join(REPO_ROOT, "dataset", "inpainting_dataset",
                                                    "ground_truth_train", "abnormal"),
                   help="フィット対象の参照マスク（学習側に使ってよい train 由来）")
    p.add_argument("--source_dir", default=os.path.join(REPO_ROOT, "dataset", "anomaly_dataset",
                                                        "train", "good"),
                   help="サンプル画像生成に使う正常画像フォルダ（読み取り専用）")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--n_pseudo", type=int, default=2000, help="変種ごとの測定枚数")
    p.add_argument("--n_samples", type=int, default=6, help="見た目確認に並べる枚数")
    p.add_argument("--imagesize", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map", "shape_stats"))
    return p.parse_args()


def _components(mask, imagesize):
    """二値マスクから欠陥ごとの形状指標を返す。"""
    lab, k = ndimage.label(mask)
    rows = []
    for i in range(1, k + 1):
        c = lab == i
        a = int(c.sum())
        if a < 4:  # ノイズ成分は除外（マスク差分由来の1-2px を拾わない）
            continue
        ys, xs = np.nonzero(c)
        h = int(ys.max() - ys.min() + 1)
        w = int(xs.max() - xs.min() + 1)
        ci = c.astype(np.int8)
        per = int(np.abs(np.diff(ci, axis=0)).sum() + np.abs(np.diff(ci, axis=1)).sum())
        rows.append({
            "area_pct": a / (imagesize * imagesize) * 100.0,
            "extent": a / (h * w),
            "circularity": 4 * np.pi * a / max(per, 1) ** 2,
            "aspect": max(h, w) / max(min(h, w), 1),
        })
    return rows, k


def measure_masks(paths, imagesize):
    rows, counts = [], []
    for p in paths:
        m = np.asarray(PIL.Image.open(p).convert("L").resize(
            (imagesize, imagesize), PIL.Image.NEAREST)) > 127
        r, k = _components(m, imagesize)
        rows.extend(r)
        counts.append(k)
    return rows, counts


def measure_variant(cfg, n, imagesize, seed, source):
    rows, counts = [], []
    for i in range(n):
        img = source.copy()
        _, mask = PA._blob(img, np.random.default_rng([seed, i]), cfg)
        r, k = _components(mask > 127, imagesize)
        rows.extend(r)
        counts.append(k)
    return rows, counts


def summarize(name, rows, counts):
    a = {k: np.array([r[k] for r in rows]) for k, _ in METRICS}
    out = {"name": name, "n_images": len(counts), "n_defects": len(rows),
           "count_median": float(np.median(counts)), "count_mean": float(np.mean(counts))}
    for k, _ in METRICS:
        out[f"{k}_p10"] = float(np.percentile(a[k], 10))
        out[f"{k}_median"] = float(np.median(a[k]))
        out[f"{k}_p90"] = float(np.percentile(a[k], 90))
    return out, a


def main():
    args = parse_args()
    run_dir = runlog.make_run_dir(args.out)
    runlog.start_tee(run_dir)

    srcs = sorted(glob.glob(os.path.join(args.source_dir, "*")))
    if not srcs:
        raise SystemExit(f"正常画像が見つかりません: {args.source_dir}")
    base = np.asarray(PIL.Image.open(srcs[0]).convert("RGB").resize(
        (args.imagesize, args.imagesize), PIL.Image.BILINEAR), dtype=np.uint8)

    groups, arrays = [], {}
    for label, d in (("実欠陥 test GT(参照)", args.real_gt), ("inpainting train GT(フィット対象)", args.fit_gt)):
        paths = sorted(glob.glob(os.path.join(d, "*")))
        if not paths:
            print(f"警告: マスクが無いのでスキップ: {d}")
            continue
        rows, counts = measure_masks(paths, args.imagesize)
        s, a = summarize(label, rows, counts)
        groups.append(s)
        arrays[label] = a

    for v in args.variants:
        cfg = VARIANTS[v]
        rows, counts = measure_variant(cfg, args.n_pseudo, args.imagesize, args.seed, base)
        s, a = summarize(v, rows, counts)
        groups.append(s)
        arrays[v] = a

    # --- CSV ---
    csv_path = os.path.join(run_dir, "shape_stats.csv")
    cols = ["name", "n_images", "n_defects", "count_median", "count_mean"] + \
        [f"{k}_{q}" for k, _ in METRICS for q in ("p10", "median", "p90")]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for g in groups:
            f.write(",".join(str(g[c]) for c in cols) + "\n")
    print(f"形状統計: {csv_path}")
    head = f"{'条件':34s}" + "".join(f"{lab:>16s}" for _, lab in METRICS)
    print(head)
    for g in groups:
        print(f"{g['name']:34s}" + "".join(f"{g[k + '_median']:16.3f}" for k, _ in METRICS))

    # --- 図1：指標ごとの分布（CDF）---
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.8))
    for ax, (k, lab) in zip(axes, METRICS):
        for name, a in arrays.items():
            v = np.sort(a[k])
            style = "--" if "GT" in name else "-"
            ax.plot(v, np.linspace(0, 1, len(v)), style, lw=1.8, label=name)
        ax.set_xlabel(lab)
        ax.set_ylabel("累積割合")
        ax.grid(alpha=0.3)
        if k == "area_pct":
            ax.set_xscale("log")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("疑似異常の形状分布 vs 実欠陥（破線＝実データ参照 / 実線＝疑似）", fontsize=12)
    fig.tight_layout()
    f1 = os.path.join(run_dir, "shape_distributions.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    print(f"図1: {f1}")

    # --- 図2：見た目の確認（各変種のサンプル）---
    n = args.n_samples
    fig, axes = plt.subplots(len(args.variants), n, figsize=(2.1 * n, 2.3 * len(args.variants)))
    axes = np.atleast_2d(axes)
    for r, v in enumerate(args.variants):
        for c in range(n):
            img, _ = PA._blob(base.copy(), np.random.default_rng([args.seed + 1, c]), VARIANTS[v])
            axes[r, c].imshow(img)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(v, fontsize=9)
    fig.suptitle("形状バリアントの見た目（同一正常画像・同一乱数列）", fontsize=12)
    fig.tight_layout()
    f2 = os.path.join(run_dir, "shape_samples.png")
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    print(f"図2: {f2}")

    runlog.write_meta(run_dir, extra={
        "real_gt": args.real_gt, "fit_gt": args.fit_gt, "variants": args.variants,
        "n_pseudo": args.n_pseudo, "seed": args.seed, "imagesize": args.imagesize,
        "blob_defaults": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in PA.DEFAULT_BLOB_CFG.items()},
        "summary": groups,
    })
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
