#!/usr/bin/env python3
"""教師 PatchCore モデルの蒸留によるメモリ削減 — 手法比較ハーネス。

round_20 等の教師モデルを読み込み、そのメモリバンク（faiss IndexFlat）を様々な
軽量表現へ置き換える手法を、同一基準（AUROC × メモリ × FPS）で比較する。

手法:
  M0  教師そのもの（厳密 L2、基準）
  M1  PQ量子化（IVFPQ）
  M2  コアセット再圧縮（greedy 再サンプリング）
  M3  生徒 MLP への距離回帰蒸留（定数サイズ・faiss バンク不要）
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))

from patchcore.runlog import Tee, jst_timestamp, make_run_dir, write_meta

import patchcore.common
import patchcore.datasets.mvtec as mvtec
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
from patchcore.distill import extract_bank_vectors
from patchcore.distill.scorers import CoresetScorer, FlatScorer, PQScorer
from patchcore.distill.student import StudentScorer
from patchcore.distill.train_student import train_student
from patchcore.distill.prototype import PrototypeScorer, train_prototype

ALL_METHODS = ["M0", "M1", "M2", "M3", "M3L", "M3R", "M4"]


def get_memory_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def embed_test_set(pc, loader, device):
    """テストセットを1度だけ埋め込み、画像毎のパッチ特徴量 [num_patches, D] を返す。

    全手法で共有して公平に比較するためキャッシュする。メモリ節約のため float16 で保持。
    """
    pc.forward_modules.eval()
    feats_per_image, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(torch.float).to(device)
            labels.extend(batch["is_anomaly"].numpy().tolist())
            features = np.asarray(pc._embed(imgs))  # [bs*num_patches, D]
            bs = imgs.shape[0]
            num_patches = features.shape[0] // bs
            features = features.reshape(bs, num_patches, -1)
            for i in range(bs):
                feats_per_image.append(features[i].astype(np.float16))
    embed_time = time.time() - t0
    return feats_per_image, np.array(labels), embed_time


def evaluate_scorer(scorer, feats_per_image, labels, embed_time):
    """各画像のパッチスコアを max 集約して画像レベル AUROC/F1 を計算する。"""
    t0 = time.time()
    image_scores = []
    for feats in feats_per_image:
        patch_scores = scorer.score_patches(feats.astype(np.float32))
        image_scores.append(float(np.max(patch_scores)))
    score_time = time.time() - t0

    m = patchcore.metrics.compute_imagewise_retrieval_metrics(image_scores, labels)
    fps = len(feats_per_image) / (embed_time + score_time)
    return {
        "method": scorer.name,
        "auroc": m["auroc"],
        "f1": m["f1"],
        "memory_mb": scorer.memory_mb(),
        "fps": fps,
        "score_time_s": score_time,
    }


def _build_parser():
    parser = argparse.ArgumentParser(description="PatchCore 蒸留手法の比較")
    parser.add_argument("teacher", help="教師モデルのフォルダ（patchcore_params.pkl を含む）")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_path", default=os.path.join(SCRIPT_DIR, "..", "dataset"))
    parser.add_argument("--classname", default="anomaly_dataset")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--no_faiss_on_gpu", action="store_true", help="faiss を CPU で動かす")
    parser.add_argument("--max_test", type=int, default=None, help="評価に使うテスト画像数の上限")
    parser.add_argument("--bank_limit", type=int, default=None, help="使うバンクベクトル数の上限（スモーク用）")
    # M1
    parser.add_argument("--pq_m", type=int, default=64)
    parser.add_argument("--pq_nbits", type=int, default=8)
    # M2
    parser.add_argument("--coreset_p", type=float, default=0.10)
    # M3 / M3L / M3R
    parser.add_argument("--student_steps", type=int, default=3000)
    parser.add_argument("--student_batch", type=int, default=4096)
    # M4（プロトタイプ）
    parser.add_argument("--proto_p", type=float, default=0.01, help="中心初期化に使うコアセット率")
    parser.add_argument("--proto_steps", type=int, default=2000)
    parser.add_argument("--proto_temp", type=float, default=1.0, help="soft-min 温度（小さいほど min に近い）")
    parser.add_argument("--out", default=os.path.join(SCRIPT_DIR, "..", "results", "Distillation"))
    return parser


def _run(args, run_dir):
    on_gpu = not args.no_faiss_on_gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  faiss_on_gpu={on_gpu}")

    # --- 教師モデルのロード ---
    nn_method = patchcore.common.FaissNN(on_gpu, args.num_workers)
    pc = patchcore.patchcore.PatchCore(device)
    pc.load_from_path(args.teacher, device=device, nn_method=nn_method)
    n_nn = pc.anomaly_scorer.n_nearest_neighbours
    print(f"教師ロード完了: {args.teacher}  (n_nn={n_nn})")

    bank = extract_bank_vectors(pc.anomaly_scorer.nn_method.search_index)
    if args.bank_limit:
        bank = bank[: args.bank_limit]
    print(f"バンクベクトル: {bank.shape[0]} × {bank.shape[1]}")

    # --- テストセット（継続学習と同設定） ---
    test_dataset = mvtec.MVTecDataset(
        args.dataset_path,
        classname=args.classname,
        resize=args.resize,
        imagesize=args.imagesize,
        split=mvtec.DatasetSplit.TEST,
        seed=args.seed,
    )
    if args.max_test:
        test_dataset = Subset(test_dataset, list(range(min(args.max_test, len(test_dataset)))))
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"テスト画像: {len(test_dataset)} 枚 → 埋め込み中...")
    feats_per_image, labels, embed_time = embed_test_set(pc, test_loader, device)
    print(f"埋め込み完了: {embed_time:.1f}s  (anomaly={int(labels.sum())}/{len(labels)})")

    # --- スコアラ構築 ---
    scorers = []
    _label_index = {"idx": None}  # M3 系/M4 の教師ラベル付け用 IndexFlatL2 を遅延生成・共有

    def ensure_label_index():
        if _label_index["idx"] is None:
            _label_index["idx"] = FlatScorer(bank, n_nn=n_nn, on_gpu=on_gpu).index
        return _label_index["idx"]

    # 生徒系の変種（M3=現行 / M3L=LayerNorm / M3R=LayerNorm+ランキング損失）
    student_variants = {
        "M3": dict(norm="batch", loss="huber", name="M3_student"),
        "M3L": dict(norm="layer", loss="huber", name="M3L_student_ln"),
        "M3R": dict(norm="layer", loss="ranking", name="M3R_student_rank"),
    }

    for method in [m for m in ALL_METHODS if m in args.methods]:
        if method == "M0":
            s = FlatScorer(bank, n_nn=n_nn, on_gpu=on_gpu)
            _label_index["idx"] = s.index
            scorers.append(s)
        elif method == "M1":
            scorers.append(
                PQScorer(bank, n_nn=n_nn, on_gpu=on_gpu, m=args.pq_m, nbits=args.pq_nbits)
            )
        elif method == "M2":
            sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(args.coreset_p, device)
            scorers.append(CoresetScorer(bank, sampler, n_nn=n_nn, on_gpu=on_gpu))
        elif method in student_variants:
            cfg = student_variants[method]
            print(f"生徒 MLP を蒸留学習中... ({method}: norm={cfg['norm']}, loss={cfg['loss']})")
            model = train_student(
                bank,
                ensure_label_index(),
                device,
                n_steps=args.student_steps,
                batch=args.student_batch,
                seed=args.seed,
                norm=cfg["norm"],
                loss=cfg["loss"],
            )
            scorers.append(StudentScorer(model, device, name=cfg["name"]))
        elif method == "M4":
            print(f"プロトタイプ中心を初期化中... (coreset_p={args.proto_p})")
            sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(args.proto_p, device)
            init_centers = np.ascontiguousarray(sampler.run(bank), dtype=np.float32)
            print(f"プロトタイプ {len(init_centers)} 個を微調整中...")
            model = train_prototype(
                bank,
                ensure_label_index(),
                init_centers,
                device,
                temp=args.proto_temp,
                n_steps=args.proto_steps,
                batch=args.student_batch,
                seed=args.seed,
            )
            scorers.append(PrototypeScorer(model, device))

    # --- 評価 ---
    rows = []
    for scorer in scorers:
        r = evaluate_scorer(scorer, feats_per_image, labels, embed_time)
        rows.append(r)
        print(
            f"  {r['method']:<22}  AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}  "
            f"mem={r['memory_mb']:.1f}MB  fps={r['fps']:.1f}"
        )

    # --- 結果保存 ---
    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w") as f:
        f.write("method,auroc,f1,memory_mb,fps,score_time_s,embed_time_s,n_test\n")
        for r in rows:
            f.write(
                f"{r['method']},{r['auroc']:.6f},{r['f1']:.6f},{r['memory_mb']:.3f},"
                f"{r['fps']:.3f},{r['score_time_s']:.3f},{embed_time:.3f},{len(labels)}\n"
            )
    print(f"\n完了  CSV: {csv_path}")
    return csv_path


def _make_plot(run_dir):
    """compare_distillation.py を呼び、規約どおり実験結果に図を伴わせる。"""
    script = os.path.join(SCRIPT_DIR, "..", "scripts", "compare_distillation.py")
    try:
        out = subprocess.run(
            [sys.executable, script, run_dir], capture_output=True, text=True
        )
        if out.stdout:
            print(out.stdout, end="")
        if out.returncode != 0:
            print(f"[warn] プロット生成に失敗: {out.stderr.strip()}")
    except Exception as e:
        print(f"[warn] プロット生成をスキップ: {e}")


def main():
    args = _build_parser().parse_args()
    run_dir = make_run_dir(args.out, jst_timestamp())
    with Tee(run_dir):
        _run(args, run_dir)
        _make_plot(run_dir)
    write_meta(run_dir, vars(args))
    print(f"[meta] {os.path.join(run_dir, 'meta.json')}")


if __name__ == "__main__":
    main()
