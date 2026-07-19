"""正常画像から疑似異常画像を生成して保存する（設計書 §12.5/§13）。

正常 train 画像をソースに、合成欠陥を載せた画像とマスクを
``results/distill_map/pseudo/<timestamp>/{anomaly,masks}/`` に保存する。
保存した画像は後段で Teacher precompute（``--image_dir``）→ Student 蒸留学習に使う。
保護対象（./dataset, /disk/data_area4）には書き込まない。
"""
import argparse
import functools
import glob
import multiprocessing as mp
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
    # blob（雑な図形描画）のサイズ・個数のランダム上限。--types blob 単独運用を想定。
    p.add_argument("--blob_count_min", type=int, default=PA.DEFAULT_BLOB_CFG["count_min"],
                   help="1画像あたり図形数の下限")
    p.add_argument("--blob_count_max", type=int, default=PA.DEFAULT_BLOB_CFG["count_max"],
                   help="1画像あたり図形数の上限")
    p.add_argument("--blob_size_min", type=float, default=PA.DEFAULT_BLOB_CFG["size_min"],
                   help="図形サイズ下限（短辺に対する比 0-1）")
    p.add_argument("--blob_size_max", type=float, default=PA.DEFAULT_BLOB_CFG["size_max"],
                   help="図形サイズ上限（短辺に対する比 0-1）")
    p.add_argument("--blob_shapes", nargs="+", default=list(PA.BLOB_SHAPES),
                   choices=list(PA.BLOB_SHAPES), help="使用する図形の集合")
    p.add_argument("--shard_size", type=int, default=5000,
                   help="1シャード(サブフォルダ)あたりの枚数。大規模で単一ディレクトリ過密を回避")
    p.add_argument("--workers", type=int, default=1, help="並列プロセス数")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map", "pseudo"))
    return p.parse_args()


def _list_images(source_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(source_dir, e)))
    return sorted(files)


def _shard_name(i, shard_size):
    return f"{i // shard_size:05d}"


def _gen_one(i, src_path, save_size, types, blob_cfg, seed):
    """index i の疑似異常を生成して (pseudo, mask, kind) を返す（rng は index 依存で再現可能）。"""
    rng = np.random.default_rng([seed, i])
    img = PIL.Image.open(src_path).convert("RGB").resize(
        (save_size, save_size), PIL.Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return PA.generate(arr, rng, types=types, blob_cfg=blob_cfg)


def _worker(arg, save_size, types, blob_cfg, seed, anom_dir, mask_dir, shard_size):
    i, src_path = arg
    pseudo, mask, kind = _gen_one(i, src_path, save_size, types, blob_cfg, seed)
    shard = _shard_name(i, shard_size)
    sid = f"pseudo_{i:08d}"
    PIL.Image.fromarray(pseudo).save(os.path.join(anom_dir, shard, sid + ".png"))
    PIL.Image.fromarray(mask).save(os.path.join(mask_dir, shard, sid + ".png"))
    return kind


def main():
    args = parse_args()
    run_dir = runlog.make_run_dir(args.out)
    runlog.start_tee(run_dir)
    anom_dir = os.path.join(run_dir, "anomaly")
    mask_dir = os.path.join(run_dir, "masks")

    sources = _list_images(args.source_dir)
    if not sources:
        raise RuntimeError(f"正常画像が見つかりません: {args.source_dir}")
    print(f"ソース正常画像: {len(sources)} 枚 → 疑似異常 {args.num} 枚生成 "
          f"(shard_size={args.shard_size}, workers={args.workers})")

    blob_cfg = {
        "count_min": args.blob_count_min, "count_max": args.blob_count_max,
        "size_min": args.blob_size_min, "size_max": args.blob_size_max,
        "shapes": tuple(args.blob_shapes),
    }
    types = tuple(args.types)

    # シャード用サブフォルダを事前作成（worker 内の makedirs 連打を避ける）
    n_shards = (args.num + args.shard_size - 1) // args.shard_size
    for s in range(n_shards):
        os.makedirs(os.path.join(anom_dir, f"{s:05d}"), exist_ok=True)
        os.makedirs(os.path.join(mask_dir, f"{s:05d}"), exist_ok=True)

    master = np.random.default_rng(args.seed)
    pick = master.integers(0, len(sources), size=args.num)
    tasks = ((i, sources[int(pick[i])]) for i in range(args.num))
    work = functools.partial(_worker, save_size=args.save_size, types=types,
                             blob_cfg=blob_cfg, seed=args.seed, anom_dir=anom_dir,
                             mask_dir=mask_dir, shard_size=args.shard_size)

    type_counts = {}
    log_every = max(1, args.num // 100)
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for k, kind in enumerate(pool.imap_unordered(work, tasks, chunksize=256)):
                type_counts[kind] = type_counts.get(kind, 0) + 1
                if (k + 1) % log_every == 0:
                    print(f"  {k + 1}/{args.num} 生成", flush=True)
    else:
        for k, arg in enumerate(tasks):
            kind = work(arg)
            type_counts[kind] = type_counts.get(kind, 0) + 1
            if (k + 1) % log_every == 0:
                print(f"  {k + 1}/{args.num} 生成", flush=True)

    # サニティ図（先頭8件を再生成。src は正常, 生成物は per-index rng で再現）
    samples = []
    for i in range(min(8, args.num)):
        sp = sources[int(pick[i])]
        orig = np.asarray(PIL.Image.open(sp).convert("RGB").resize(
            (args.save_size, args.save_size), PIL.Image.BILINEAR), dtype=np.uint8)
        pseudo, mask, kind = _gen_one(i, sp, args.save_size, types, blob_cfg, args.seed)
        samples.append((orig, pseudo, mask, kind))
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
        "blob_cfg": blob_cfg, "shard_size": args.shard_size, "n_shards": n_shards,
        "workers": args.workers,
    })
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
