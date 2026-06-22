"""正常画像から疑似異常画像を生成して保存する（設計書 §12.5/§13）。

正常 train 画像をソースに、合成欠陥を載せた画像とマスクを
``results/distill_map/pseudo/<timestamp>/{anomaly,masks}/`` に保存する。
保存した画像は後段で Teacher precompute（``--image_dir``）→ Student 蒸留学習に使う。
保護対象（./dataset, /disk/data_area4）には書き込まない。
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import patchcore.runlog as runlog
from patchcore.distill_map import pseudo_anomaly as PA

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"


def parse_args():
    p = argparse.ArgumentParser(description="疑似異常画像の生成")
    p.add_argument("--source_dir",
                   default=os.path.join(REPO_ROOT, "dataset", "anomaly_dataset", "train", "good"),
                   help="正常画像フォルダ（読み取り専用）")
    p.add_argument("--num", type=int, default=1000, help="生成枚数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_size", type=int, default=256, help="保存解像度（precompute の resize に合わせる）")
    p.add_argument("--types", nargs="+", default=list(PA.ANOMALY_TYPES), choices=list(PA.ANOMALY_TYPES))
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map", "pseudo"))
    return p.parse_args()


def _list_images(source_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(source_dir, e)))
    return sorted(files)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    run_dir = runlog.make_run_dir(args.out)
    runlog.start_tee(run_dir)
    anom_dir = os.path.join(run_dir, "anomaly")
    mask_dir = os.path.join(run_dir, "masks")
    os.makedirs(anom_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    sources = _list_images(args.source_dir)
    if not sources:
        raise RuntimeError(f"正常画像が見つかりません: {args.source_dir}")
    print(f"ソース正常画像: {len(sources)} 枚 → 疑似異常 {args.num} 枚生成")

    pick = rng.integers(0, len(sources), size=args.num)
    type_counts = {}
    samples = []
    for i, idx in enumerate(pick):
        img = PIL.Image.open(sources[idx]).convert("RGB").resize(
            (args.save_size, args.save_size), PIL.Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.uint8)
        pseudo, mask, kind = PA.generate(arr, rng, types=tuple(args.types))
        type_counts[kind] = type_counts.get(kind, 0) + 1
        sid = f"pseudo_{i:06d}"
        PIL.Image.fromarray(pseudo).save(os.path.join(anom_dir, sid + ".png"))
        PIL.Image.fromarray(mask).save(os.path.join(mask_dir, sid + ".png"))
        if len(samples) < 8:
            samples.append((arr, pseudo, mask, kind))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{args.num} 生成", flush=True)

    # サニティ図
    fig, axes = plt.subplots(len(samples), 3, figsize=(9, 3 * len(samples)))
    if len(samples) == 1:
        axes = axes[None, :]
    for r, (orig, pseudo, mask, kind) in enumerate(samples):
        axes[r, 0].imshow(orig); axes[r, 0].set_title("正常")
        axes[r, 1].imshow(pseudo); axes[r, 1].set_title(f"疑似異常: {kind}")
        axes[r, 2].imshow(mask, cmap="gray"); axes[r, 2].set_title("マスク")
        for c in range(3):
            axes[r, c].set_xlabel("x [px]"); axes[r, c].set_ylabel("y [px]")
    fig.suptitle("疑似異常生成サニティ", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "pseudo_sanity.png"), dpi=150)
    plt.close(fig)

    print(f"種別内訳: {type_counts}")
    runlog.write_meta(run_dir, extra={
        "source_dir": args.source_dir, "num": args.num, "seed": args.seed,
        "save_size": args.save_size, "types": args.types, "type_counts": type_counts,
    })
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
