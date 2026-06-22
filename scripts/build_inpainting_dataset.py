#!/usr/bin/env python3
"""Inpainting の original/anomaly ペアを MVTec 形式データセットに再構成する。

入力（展開済み）:
  <test_src>/NNNNN_original.jpg, NNNNN_anomaly.jpg
  <train_src>/NNNNN_original.jpg, NNNNN_anomaly.jpg

出力（MVTec 形式, anomaly_dataset を踏襲）:
  <out>/train/good/        train_normal_NNNNN.jpg      (= train *_original)
  <out>/train/abnormal/    train_abnormal_NNNNN.jpg    (= train *_anomaly)
  <out>/test/good/         test_normal_NNNNN.jpg       (= test  *_original)
  <out>/test/abnormal/     test_abnormal_NNNNN.jpg     (= test  *_anomaly)
  <out>/ground_truth/abnormal/        test_abnormal_NNNNN_mask.jpg   (loader 対応)
  <out>/ground_truth_train/abnormal/  train_abnormal_NNNNN_mask.jpg  (参考用・loader 非対象)

マスクは original↔anomaly のグレースケール差分を二値化して生成する
（白=異常, PNG 1bit。参考 anomaly_dataset と同形式）。
original が欠損している anomaly はマスクを生成しない（画像のみコピー）。
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MVTec-style dataset from Inpainting pairs")
    p.add_argument("--test-src", type=Path, required=True, help="展開済み Inpainting_test ディレクトリ")
    p.add_argument("--train-src", type=Path, required=True, help="展開済み Inpainting_train ディレクトリ")
    p.add_argument("--out", type=Path, required=True, help="出力データセットルート")
    p.add_argument("--threshold", type=int, default=30, help="差分の二値化しきい値 (0-255)")
    p.add_argument("--min-area", type=int, default=16, help="この画素数未満の連結成分は除去")
    p.add_argument("--overwrite", action="store_true", help="既存出力を許容（上書き）")
    return p.parse_args()


def make_mask(original: Path, anomaly: Path, threshold: int, min_area: int) -> Image.Image:
    o = np.asarray(Image.open(original).convert("L"), dtype=np.int16)
    a = np.asarray(Image.open(anomaly).convert("L"), dtype=np.int16)
    diff = np.abs(a - o)
    mask = diff > threshold
    mask = ndimage.binary_closing(mask, iterations=1)
    mask = ndimage.binary_fill_holes(mask)
    if min_area > 0 and mask.any():
        labels, n = ndimage.label(mask)
        if n > 0:
            sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
            keep = {i + 1 for i, s in enumerate(sizes) if s >= min_area}
            mask = np.isin(labels, list(keep)) if keep else np.zeros_like(mask)
    return Image.fromarray(mask.astype(bool)).convert("1")


def stem_id(path: Path, suffix: str) -> str:
    """'00001_original.jpg' -> '00001'"""
    return path.name[: -len(suffix)]


def process_split(src: Path, out: Path, split: str, gt_dir: Path,
                  threshold: int, min_area: int) -> dict:
    good_dir = out / split / "good"
    abn_dir = out / split / "abnormal"
    for d in (good_dir, abn_dir, gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    originals = {stem_id(p, "_original.jpg"): p for p in sorted(src.glob("*_original.jpg"))}
    anomalies = {stem_id(p, "_anomaly.jpg"): p for p in sorted(src.glob("*_anomaly.jpg"))}

    counts = {"good": 0, "abnormal": 0, "mask": 0, "mask_skipped": 0}

    for num, op in originals.items():
        shutil.copy2(op, good_dir / f"{split}_normal_{num}.jpg")
        counts["good"] += 1

    for num, ap in anomalies.items():
        shutil.copy2(ap, abn_dir / f"{split}_abnormal_{num}.jpg")
        counts["abnormal"] += 1
        if num in originals:
            mask = make_mask(originals[num], ap, threshold, min_area)
            mask.save(gt_dir / f"{split}_abnormal_{num}_mask.jpg", format="PNG")
            counts["mask"] += 1
        else:
            counts["mask_skipped"] += 1
    return counts


def main() -> int:
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        print(f"Error: 出力先が既に存在します: {args.out} (--overwrite で許可)")
        return 1

    test_counts = process_split(
        args.test_src, args.out, "test",
        args.out / "ground_truth" / "abnormal",
        args.threshold, args.min_area,
    )
    train_counts = process_split(
        args.train_src, args.out, "train",
        args.out / "ground_truth_train" / "abnormal",
        args.threshold, args.min_area,
    )

    print(f"[test]  {test_counts}")
    print(f"[train] {train_counts}")
    print(f"Done. -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
