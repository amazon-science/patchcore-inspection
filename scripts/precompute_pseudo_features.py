#!/usr/bin/env python3
"""疑似異常 packed の Teacher ResNet50 特徴マップ F_T を precompute し packed 保存する。

②(feature_to_map) の入力・③(image_to_feature) の学習ターゲット・定量評価表の
「特徴map評価(feat_MSE/MAE)」列に必要な **Teacher 特徴 F_T([C,H',W'] fp16)** を、
既存の packed（B方式の ``<run>/pseudo``。``images.npy`` と ``targets/scores/labels``
を持つ）に対して計算する。

- 特徴抽出は **NN(faiss) 不要 = backbone forward のみ**なので B方式（距離マップ生成,
  ~5 img/s）よりはるかに速い。
- 出力は per-file ``.pt`` を作らず **packed memmap ``features.npy``**（大規模向け）。
  ``PackedTargetDataset(load_feature=True)`` がそのまま読める。
- ``images/targets/scores/labels`` は入力 pack から **symlink 共有**し、``features.npy``
  と ``done.npy`` のみ新規作成 → 学習は ``--pseudo_dir <out>`` を渡すだけで
  画像・異常map正解・特徴の全てが揃う。
- ``--start/--end`` で範囲分割（複数GPU並列）、``done.npy`` で中断復帰。

CLAUDE.md 準拠：出力先は明示指定のみ、meta.json / log.txt を残す。
大容量データは ``/disk/data_area4/t-hori/bfai/`` に置く運用（ユーザー方針）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import patchcore.common
import patchcore.patchcore
import patchcore.runlog as runlog
from patchcore.distill_map.io_utils import IMAGENET_MEAN, IMAGENET_STD, PACK_MARKER

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARE = ("images.npy", "targets.npy", "scores.npy", "labels.npy")


def parse_args():
    p = argparse.ArgumentParser(description="疑似 packed の Teacher 特徴 F_T を precompute（packed）")
    p.add_argument("teacher", help="教師モデルのフォルダ（patchcore_params.pkl を含む）")
    p.add_argument("--pack", required=True, help="入力 packed（B方式の <run>。images.npy を持つ）")
    p.add_argument("--out", required=True, help="出力 packed（新規。features.npy を作り他は symlink 共有）")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8, help="faiss スレッド数（未使用だが構築に必要）")
    p.add_argument("--no_faiss_on_gpu", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=0, help="0=最後まで")
    p.add_argument("--init_only", action="store_true",
                   help="feature_shape 検出と割り当てだけ行って終了（並列実行前に1回）")
    p.add_argument("--limit", type=int, default=0, help=">0 でこの枚数だけ処理（スモーク用）")
    p.add_argument("--log_every", type=int, default=2000)
    return p.parse_args()


def _load_teacher(args, device):
    nn_method = patchcore.common.FaissNN(not args.no_faiss_on_gpu, args.num_workers)
    pc = patchcore.patchcore.PatchCore(device)
    pc.load_from_path(args.teacher, device=device, nn_method=nn_method)
    return pc


def _embed_grid(pc, x):
    """正規化済み入力 x([B,3,H,W]) → Teacher 特徴格子 [B,C,H',W']（NN 不要）。"""
    images = x.to(torch.float).to(pc.device)
    _ = pc.forward_modules.eval()
    b = images.shape[0]
    with torch.no_grad():
        features, patch_shapes = pc._embed(images, provide_patch_shapes=True)
    features = np.asarray(features)          # [B*P, C]
    h, w = patch_shapes[0]
    c = features.shape[-1]
    return features.reshape(b, h, w, c).transpose(0, 3, 1, 2)  # [B,C,H',W']


def _init_pack(args, src_meta, feature_shape):
    """features.npy / done.npy を割り当て、他配列は入力 pack から symlink・meta を書く。"""
    out = args.out
    os.makedirs(out, exist_ok=True)
    count = int(src_meta["count"])
    for name in _SHARE:
        src = os.path.abspath(os.path.join(args.pack, name))
        dst = os.path.join(out, name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    feat_path = os.path.join(out, "features.npy")
    if not os.path.exists(feat_path):
        np.lib.format.open_memmap(feat_path, mode="w+", dtype=np.float16,
                                  shape=(count, *feature_shape)).flush()
    done_path = os.path.join(out, "done.npy")
    if not os.path.exists(done_path):
        np.lib.format.open_memmap(done_path, mode="w+", dtype=np.uint8, shape=(count,)).flush()
    meta_path = os.path.join(out, PACK_MARKER)
    meta = dict(src_meta)
    meta.update({
        "feature_shape": [int(v) for v in feature_shape],
        "feature_source": "make_pseudo_features(teacher_resnet)",
        "src_pack": os.path.abspath(args.pack),
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return count


def main():
    args = parse_args()
    with open(os.path.join(args.pack, PACK_MARKER)) as f:
        src_meta = json.load(f)
    if not src_meta.get("save_image"):
        raise SystemExit(f"入力 pack に images.npy がありません: {args.pack}")
    count = int(src_meta["count"])

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    imgs = np.load(os.path.join(args.pack, "images.npy"), mmap_mode="r")

    t0 = time.time()
    pc = _load_teacher(args, device)
    print(f"教師ロード完了: {args.teacher} ({time.time() - t0:.1f}s)", flush=True)

    # feature_shape を実 forward で検出（1枚）
    a0 = np.array(imgs[0:1], dtype=np.float32) / 255.0
    g0 = _embed_grid(pc, (torch.from_numpy(a0) - mean) / std)
    feature_shape = list(g0.shape[1:])          # [C,H',W']
    print(f"feature_shape={feature_shape}  (1枚={np.prod(feature_shape)*2/1e6:.2f}MB, "
          f"全{count}枚={np.prod(feature_shape)*2*count/1e9:.1f}GB)", flush=True)

    _init_pack(args, src_meta, feature_shape)
    if args.init_only:
        print(f"割り当て完了: {args.out}（count={count}, feature_shape={feature_shape}）")
        return

    runlog.start_tee(args.out)
    end = args.end or count
    if args.limit:
        end = min(end, args.start + args.limit)
    print(f"対象範囲: [{args.start}, {end}) / 全 {count}  gpu={args.gpu} batch={args.batch_size}")

    feat = np.load(os.path.join(args.out, "features.npy"), mmap_mode="r+")
    done = np.load(os.path.join(args.out, "done.npy"), mmap_mode="r+")

    n_done = 0
    t_start = time.time()
    i = args.start
    while i < end:
        j = min(i + args.batch_size, end)
        idx = [k for k in range(i, j) if not done[k]]
        if not idx:
            i = j
            continue
        a = np.array(imgs[idx], dtype=np.float32) / 255.0
        grid = _embed_grid(pc, (torch.from_numpy(a) - mean) / std)
        for m, k in enumerate(idx):
            feat[k] = np.asarray(grid[m], dtype=np.float16)
            done[k] = 1
        n_done += len(idx)
        if n_done and n_done % args.log_every < args.batch_size:
            el = time.time() - t_start
            fps = n_done / max(el, 1e-9)
            rem = (end - j) / max(fps, 1e-9) / 3600
            print(f"  {j - args.start}/{end - args.start} 完了  {fps:.1f} img/s  残り約 {rem:.2f}h",
                  flush=True)
        i = j

    feat.flush()
    done.flush()
    n_all = int(np.asarray(done).sum())
    print(f"完了: この実行で {n_done} 枚 / 全体 {n_all}/{count} 済み")

    meta_path = os.path.join(args.out, PACK_MARKER)
    with open(meta_path) as f:
        meta = json.load(f)
    meta.update({"n_feat_done": n_all, "feat_complete": bool(n_all == count)})
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    runlog.write_meta(args.out, extra={
        "teacher": args.teacher, "src_pack": args.pack, "range": [args.start, end],
        "count": count, "feature_shape": feature_shape, "n_feat_done": n_all,
        "batch_size": args.batch_size,
    })
    print(f"出力: {args.out}  → 学習は --pseudo_dir {args.out}（load_feature 自動）")


if __name__ == "__main__":
    main()
