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
from patchcore.distill_map import io_utils
from patchcore.distill_map import pseudo_anomaly as PA

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"


def parse_args():
    p = argparse.ArgumentParser(description="疑似異常画像の生成")
    p.add_argument("--source_dir",
                   default=os.path.join(REPO_ROOT, "dataset", "anomaly_dataset", "train", "good"),
                   help="正常画像フォルダ（読み取り専用）")
    p.add_argument("--num", type=int, default=1000,
                   help="生成枚数（--per_image=0 のランダム抽出モード時）")
    p.add_argument("--per_image", type=int, default=0,
                   help=">0 で「正常1枚につき N 枚」を順番に生成（各ソースを1回だけ読む=I/O最小）。"
                        "総数=正常枚数×N")
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
    # 直接 packed 出力（PNG を介さず image+ぼかしターゲットを memmap へ。NFS 大規模向け）
    p.add_argument("--pack", action="store_true",
                   help="PNG でなく packed memmap を <run>/pseudo/ に直接生成（--per_image 必須）")
    p.add_argument("--resize", type=int, default=256, help="pack: 整列 resize（学習前処理に一致）")
    p.add_argument("--imagesize", type=int, default=224, help="pack: center-crop 後の保存解像度")
    p.add_argument("--mask_blur_sigma", type=float, default=2.0, help="pack: ターゲットのぼかし σ[px]")
    p.add_argument("--sample_png", type=int, default=200, help="pack: 閲覧用に保存するサンプル PNG 枚数")
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


def _load_source(src_path, save_size):
    return np.asarray(PIL.Image.open(src_path).convert("RGB").resize(
        (save_size, save_size), PIL.Image.BILINEAR), dtype=np.uint8)


def _gen_variant(arr, i, types, blob_cfg, seed):
    """index i の疑似異常を生成（rng は index 依存で並列でも再現可能）。"""
    rng = np.random.default_rng([seed, i])
    return PA.generate(arr, rng, types=types, blob_cfg=blob_cfg)


def _save_pair(i, pseudo, mask, anom_dir, mask_dir, shard_size):
    shard = _shard_name(i, shard_size)
    sid = f"pseudo_{i:08d}"
    PIL.Image.fromarray(pseudo).save(os.path.join(anom_dir, shard, sid + ".png"), compress_level=1)
    PIL.Image.fromarray(mask).save(os.path.join(mask_dir, shard, sid + ".png"), compress_level=1)


def _worker_random(arg, save_size, types, blob_cfg, seed, anom_dir, mask_dir, shard_size):
    """ランダム抽出モード：index i とソースパスから 1 枚生成。"""
    i, src_path = arg
    arr = _load_source(src_path, save_size)
    pseudo, mask, kind = _gen_variant(arr, i, types, blob_cfg, seed)
    _save_pair(i, pseudo, mask, anom_dir, mask_dir, shard_size)
    return {kind: 1}


def _worker_source(arg, per_image, save_size, types, blob_cfg, seed, anom_dir, mask_dir, shard_size):
    """ソース基点モード：正常1枚(j)を1回だけ読み、per_image 枚を連番生成。I/O 最小。"""
    j, src_path = arg
    arr = _load_source(src_path, save_size)
    counts = {}
    for k in range(per_image):
        i = j * per_image + k
        pseudo, mask, kind = _gen_variant(arr, i, types, blob_cfg, seed)
        _save_pair(i, pseudo, mask, anom_dir, mask_dir, shard_size)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


# --- 直接 packed 出力（PNG を介さず memmap へ）。_PW は親で確保し fork で子に共有 ---
_PW = None


def _resize_center_crop(im, resize, imagesize, interp):
    im = im.resize((resize, resize), interp)
    off = (resize - imagesize) // 2
    return im.crop((off, off, off + imagesize, off + imagesize))


def _to_pack(pseudo, mask, resize, imagesize, sigma):
    """(pseudo, mask) を学習前処理に一致する 224 の (image uint8[3,S,S], target fp16[S,S], score)。"""
    from scipy.ndimage import gaussian_filter
    pim = _resize_center_crop(PIL.Image.fromarray(pseudo), resize, imagesize, PIL.Image.BILINEAR)
    img = np.asarray(pim, dtype=np.uint8).transpose(2, 0, 1)
    mim = _resize_center_crop(PIL.Image.fromarray(mask), resize, imagesize, PIL.Image.NEAREST)
    marr = (np.asarray(mim, dtype=np.float32) > 127).astype(np.float32)
    if sigma and sigma > 0:
        marr = gaussian_filter(marr, sigma=sigma)
    marr = np.clip(marr, 0.0, 1.0)
    return img, marr.astype(np.float16), float(marr.max())


def _worker_source_pack(arg, per_image, save_size, resize, imagesize, types, blob_cfg, seed, sigma):
    """ソース基点＋直接 pack：正常1枚を読み per_image 枚を memmap の連番 index に書く。"""
    j, src_path = arg
    arr = _load_source(src_path, save_size)
    counts = {}
    for k in range(per_image):
        i = j * per_image + k
        pseudo, mask, kind = _gen_variant(arr, i, types, blob_cfg, seed)
        img, tgt, score = _to_pack(pseudo, mask, resize, imagesize, sigma)
        _PW.images[i] = img
        _PW.targets[i] = tgt
        _PW.scores[i] = score
        _PW.labels[i] = 1
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _run_pack(args, sources, types, blob_cfg, num, run_dir):
    """直接 packed 出力：<run>/pseudo/ に memmap を書き、閲覧用サンプル PNG とサニティ図も残す。"""
    global _PW
    pack_dir = os.path.join(run_dir, "pseudo")
    S = args.imagesize
    _PW = io_utils.PackedWriter(pack_dir, num, imagesize=S, map_size=(S, S), save_image=True)
    work = functools.partial(_worker_source_pack, per_image=args.per_image,
                             save_size=args.save_size, resize=args.resize, imagesize=S,
                             types=types, blob_cfg=blob_cfg, seed=args.seed,
                             sigma=args.mask_blur_sigma)
    tasks = list(enumerate(sources))
    type_counts = {}
    log_every = max(1, len(sources) // 100)

    def _acc(counts):
        for kk, vv in counts.items():
            type_counts[kk] = type_counts.get(kk, 0) + vv

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for u, counts in enumerate(pool.imap_unordered(work, tasks, chunksize=8)):
                _acc(counts)
                if (u + 1) % log_every == 0:
                    print(f"  {u + 1}/{len(sources)} source", flush=True)
    else:
        for u, arg in enumerate(tasks):
            _acc(work(arg))
            if (u + 1) % log_every == 0:
                print(f"  {u + 1}/{len(sources)} source", flush=True)
    _PW.close(extra_meta={"source": "generate_blob_direct", "blob_cfg": blob_cfg,
                          "blur_sigma": args.mask_blur_sigma, "per_image": args.per_image,
                          "n_sources": len(sources), "seed": args.seed})

    # 閲覧用サンプル PNG（先頭 sample_png 枚）とサニティ図
    ds = io_utils.PackedTargetDataset(pack_dir, in_memory=False)
    mean = np.array(io_utils.IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(io_utils.IMAGENET_STD).reshape(3, 1, 1)
    n_png = min(args.sample_png, num)
    if n_png > 0:
        sp_dir = os.path.join(run_dir, "sample_png")
        os.makedirs(sp_dir, exist_ok=True)
        for i in range(n_png):
            PIL.Image.fromarray(ds.images[i].transpose(1, 2, 0)).save(
                os.path.join(sp_dir, f"pseudo_{i:08d}.png"), compress_level=1)
        print(f"サンプル PNG: {n_png} 枚 → {sp_dir}")
    samples = []
    for i in range(min(8, num)):
        img = ds.images[i].transpose(1, 2, 0).astype(np.float32) / 255.0
        samples.append((img, np.asarray(ds.targets[i], dtype=np.float32)))
    fig, axes = plt.subplots(len(samples), 3, figsize=(9, 3 * len(samples)))
    if len(samples) == 1:
        axes = axes[None, :]
    for r, (img, tgt) in enumerate(samples):
        axes[r, 0].imshow(img); axes[r, 0].set_title("疑似異常")
        axes[r, 1].imshow(tgt, cmap="jet", vmin=0, vmax=1); axes[r, 1].set_title("ぼかしターゲット")
        axes[r, 2].imshow(img); axes[r, 2].imshow(tgt, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[r, 2].set_title("オーバーレイ")
        for c in range(3):
            axes[r, c].set_xlabel("x [px]"); axes[r, c].set_ylabel("y [px]")
    fig.suptitle(f"直接pack サニティ（σ={args.mask_blur_sigma}）", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "pack_sanity.png"), dpi=150)
    plt.close(fig)

    print(f"種別内訳: {type_counts}")
    runlog.write_meta(run_dir, extra={
        "source_dir": args.source_dir, "num": num, "per_image": args.per_image,
        "n_sources": len(sources), "seed": args.seed, "pack": True,
        "resize": args.resize, "imagesize": S, "mask_blur_sigma": args.mask_blur_sigma,
        "save_size": args.save_size, "types": args.types, "type_counts": type_counts,
        "blob_cfg": blob_cfg, "workers": args.workers, "pack_dir": pack_dir,
    })
    print(f"完了(pack): {run_dir}  → 学習は --pseudo_dir {pack_dir}")


def main():
    args = parse_args()
    run_dir = runlog.make_run_dir(args.out)
    runlog.start_tee(run_dir)
    anom_dir = os.path.join(run_dir, "anomaly")
    mask_dir = os.path.join(run_dir, "masks")

    sources = _list_images(args.source_dir)
    if not sources:
        raise RuntimeError(f"正常画像が見つかりません: {args.source_dir}")

    blob_cfg = {
        "count_min": args.blob_count_min, "count_max": args.blob_count_max,
        "size_min": args.blob_size_min, "size_max": args.blob_size_max,
        "shapes": tuple(args.blob_shapes),
    }
    types = tuple(args.types)
    source_major = args.per_image > 0
    num = len(sources) * args.per_image if source_major else args.num
    mode = f"ソース基点(1枚×{args.per_image})" if source_major else "ランダム抽出"
    print(f"ソース正常画像: {len(sources)} 枚 → 疑似異常 {num} 枚生成 "
          f"[{mode}{' +pack' if args.pack else ''}] (workers={args.workers})")

    if args.pack:
        if not source_major:
            raise SystemExit("--pack は --per_image>0（ソース基点）と併用してください")
        _run_pack(args, sources, types, blob_cfg, num, run_dir)
        return

    # シャード用サブフォルダを事前作成（worker 内の makedirs 連打を避ける）
    n_shards = (num + args.shard_size - 1) // args.shard_size
    for s in range(n_shards):
        os.makedirs(os.path.join(anom_dir, f"{s:05d}"), exist_ok=True)
        os.makedirs(os.path.join(mask_dir, f"{s:05d}"), exist_ok=True)

    if source_major:
        tasks = list(enumerate(sources))
        work = functools.partial(_worker_source, per_image=args.per_image,
                                 save_size=args.save_size, types=types, blob_cfg=blob_cfg,
                                 seed=args.seed, anom_dir=anom_dir, mask_dir=mask_dir,
                                 shard_size=args.shard_size)
        total_units, unit = len(sources), "source"
    else:
        master = np.random.default_rng(args.seed)
        pick = master.integers(0, len(sources), size=args.num)
        tasks = [(i, sources[int(pick[i])]) for i in range(args.num)]
        work = functools.partial(_worker_random, save_size=args.save_size, types=types,
                                 blob_cfg=blob_cfg, seed=args.seed, anom_dir=anom_dir,
                                 mask_dir=mask_dir, shard_size=args.shard_size)
        total_units, unit = args.num, "枚"

    type_counts = {}
    log_every = max(1, total_units // 100)

    def _acc(counts):
        for kk, vv in counts.items():
            type_counts[kk] = type_counts.get(kk, 0) + vv

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for u, counts in enumerate(pool.imap_unordered(work, tasks, chunksize=8)):
                _acc(counts)
                if (u + 1) % log_every == 0:
                    print(f"  {u + 1}/{total_units} {unit}", flush=True)
    else:
        for u, arg in enumerate(tasks):
            _acc(work(arg))
            if (u + 1) % log_every == 0:
                print(f"  {u + 1}/{total_units} {unit}", flush=True)

    # サニティ図（source 0 の先頭8枚を再生成）
    samples = []
    orig0 = _load_source(sources[0], args.save_size)
    for i in range(min(8, num)):
        pseudo, mask, kind = _gen_variant(orig0, i, types, blob_cfg, args.seed)
        samples.append((orig0, pseudo, mask, kind))
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
        "source_dir": args.source_dir, "num": num, "per_image": args.per_image,
        "n_sources": len(sources), "seed": args.seed,
        "save_size": args.save_size, "types": args.types, "type_counts": type_counts,
        "blob_cfg": blob_cfg, "shard_size": args.shard_size, "n_shards": n_shards,
        "workers": args.workers,
    })
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
