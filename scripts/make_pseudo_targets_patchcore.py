#!/usr/bin/env python3
"""疑似異常の正解異常マップを PatchCore（メモリバンク）で計算し packed 保存する（B方式）。

A方式（描画マスクを正解）は「教師あり学習」であって PatchCore の蒸留ではない。
B方式は疑似画像を Teacher PatchCore に通し、**メモリバンクへの最近傍距離マップ**を
正解にするため、Student は PatchCore の距離関数そのものを近似する（＝真の蒸留）。

入力は A方式で作った packed フォルダ（``images.npy`` を持つ）。画像は再生成せず
**symlink で共有**し、``targets/scores/labels`` だけを PatchCore 出力で作り直す。
出力はそのまま ``train_distill_map.py --pseudo_dir`` に渡せる。

Teacher 推論は重い（実測 約5.4 img/s）ため、
- ``--start/--end`` で範囲分割 → 複数GPUで並列実行
- ``done.npy`` で進捗を記録し、再実行時は済んだ行をスキップ（中断復帰）
に対応する。

CLAUDE.md 準拠：出力先は明示指定のみ、meta.json / log.txt を残す。
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
from patchcore.distill_map.io_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PACK_MARKER,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="疑似異常の正解を PatchCore 距離マップで作る（B方式）")
    p.add_argument("teacher", help="教師モデルのフォルダ（patchcore_params.pkl を含む）")
    p.add_argument("--pack", required=True, help="入力 packed フォルダ（A方式の <run>/pseudo）")
    p.add_argument("--out", required=True, help="出力 packed フォルダ（新規。images は symlink 共有）")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8, help="faiss のスレッド数")
    p.add_argument("--no_faiss_on_gpu", action="store_true")
    p.add_argument("--start", type=int, default=0, help="処理開始 index（並列分割用）")
    p.add_argument("--end", type=int, default=0, help="処理終了 index（0=最後まで）")
    p.add_argument("--init_only", action="store_true",
                   help="出力の割り当てだけ行って終了（並列実行前に1回だけ実施）")
    p.add_argument("--limit", type=int, default=0, help=">0 でこの枚数だけ処理（スモーク用）")
    p.add_argument("--log_every", type=int, default=2000)
    return p.parse_args()


def _init_pack(src_pack, out_dir, count, imagesize):
    """出力 packed を割り当て、images.npy は入力から symlink する。"""
    os.makedirs(out_dir, exist_ok=True)
    src_img = os.path.abspath(os.path.join(src_pack, "images.npy"))
    dst_img = os.path.join(out_dir, "images.npy")
    if not os.path.exists(dst_img):
        os.symlink(src_img, dst_img)  # 画像は共有（56GB の複製を避ける）
    for name, dtype, shape in (
        ("targets.npy", np.float16, (count, imagesize, imagesize)),
        ("scores.npy", np.float32, (count,)),
        ("labels.npy", np.int8, (count,)),
        ("done.npy", np.uint8, (count,)),
    ):
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape).flush()
    meta_path = os.path.join(out_dir, PACK_MARKER)
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump({
                "count": int(count), "imagesize": int(imagesize),
                "map_size": [int(imagesize), int(imagesize)],
                "save_image": True, "feature_shape": None,
                # 重要：正解が PatchCore 距離スケールであることを示す。
                # 学習側の --pseudo_target_scale オートはこれを見て倍率を掛けない。
                "target_space": "patchcore",
                "source": "make_pseudo_targets_patchcore",
                "src_pack": os.path.abspath(src_pack),
            }, f, ensure_ascii=False, indent=2)
    return meta_path


def main():
    args = parse_args()
    with open(os.path.join(args.pack, PACK_MARKER)) as f:
        src_meta = json.load(f)
    count = int(src_meta["count"])
    S = int(src_meta["imagesize"])
    if not src_meta.get("save_image"):
        raise SystemExit(f"入力 pack に images.npy がありません: {args.pack}")

    _init_pack(args.pack, args.out, count, S)
    if args.init_only:
        print(f"割り当て完了: {args.out}（count={count}, imagesize={S}）")
        return

    runlog.start_tee(args.out)
    end = args.end or count
    if args.limit:
        end = min(end, args.start + args.limit)
    print(f"対象範囲: [{args.start}, {end}) / 全 {count}  gpu={args.gpu} batch={args.batch_size}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    nn_method = patchcore.common.FaissNN(not args.no_faiss_on_gpu, args.num_workers)
    pc = patchcore.patchcore.PatchCore(device)
    pc.load_from_path(args.teacher, device=device, nn_method=nn_method)
    print(f"教師ロード完了: {args.teacher} ({time.time() - t0:.1f}s)", flush=True)

    imgs = np.load(os.path.join(args.pack, "images.npy"), mmap_mode="r")
    tgt = np.load(os.path.join(args.out, "targets.npy"), mmap_mode="r+")
    sco = np.load(os.path.join(args.out, "scores.npy"), mmap_mode="r+")
    lab = np.load(os.path.join(args.out, "labels.npy"), mmap_mode="r+")
    done = np.load(os.path.join(args.out, "done.npy"), mmap_mode="r+")
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    n_done = 0
    t_start = time.time()
    i = args.start
    while i < end:
        j = min(i + args.batch_size, end)
        idx = [k for k in range(i, j) if not done[k]]   # 中断復帰：済みはスキップ
        if not idx:
            i = j
            continue
        a = np.array(imgs[idx], dtype=np.float32) / 255.0
        x = (torch.from_numpy(a) - mean) / std
        scores, amaps = pc._predict(x)
        for m, k in enumerate(idx):
            tgt[k] = np.asarray(amaps[m], dtype=np.float16)
            sco[k] = float(scores[m])
            lab[k] = 1
            done[k] = 1
        n_done += len(idx)
        if n_done and n_done % args.log_every < args.batch_size:
            el = time.time() - t_start
            fps = n_done / max(el, 1e-9)
            rem = (end - j) / max(fps, 1e-9) / 3600
            print(f"  {j - args.start}/{end - args.start} 完了  {fps:.1f} img/s  残り約 {rem:.2f}h",
                  flush=True)
        i = j

    for a in (tgt, sco, lab, done):
        a.flush()
    n_all = int(np.asarray(done).sum())
    s = np.asarray(sco)[np.asarray(done) == 1].astype(np.float64)
    print(f"完了: この実行で {n_done} 枚 / 全体 {n_all}/{count} 済み")
    if len(s):
        print(f"PatchCore スコア統計: mean={s.mean():.3f} min={s.min():.3f} max={s.max():.3f}")

    # 全部終わったら meta にスコア統計を追記（学習側は scores.npy を直接読むので任意）
    meta_path = os.path.join(args.out, PACK_MARKER)
    with open(meta_path) as f:
        meta = json.load(f)
    meta.update({"n_done": n_all, "complete": bool(n_all == count)})
    if len(s):
        meta.update({"score_mean": float(s.mean()), "score_std": float(s.std()),
                     "score_min": float(s.min()), "score_max": float(s.max()),
                     "n_anomaly": int(n_all)})
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    runlog.write_meta(args.out, extra={
        "teacher": args.teacher, "src_pack": args.pack, "range": [args.start, end],
        "count": count, "n_done_total": n_all, "batch_size": args.batch_size,
    })
    print(f"出力: {args.out}  → 学習は --pseudo_dir {args.out}")


if __name__ == "__main__":
    main()
