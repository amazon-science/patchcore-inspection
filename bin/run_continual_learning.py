#!/usr/bin/env python3
"""
継続学習実験: PatchCore
各ラウンドで異なる N_PER_ROUND 枚を学習し、コアセットをマージしていく。
"""
import argparse
import gc
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))

import patchcore.backbones
import patchcore.common
import patchcore.datasets.mvtec as mvtec
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler

# ---------- 設定 ----------
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "dataset")        # データセットのルートディレクトリ
RESULTS_PATH = os.path.join(SCRIPT_DIR, "..", "results", "ContinualLearning")  # 結果・チェックポイントの保存先
MEMORY_LIMIT_MB = 48_000    # プロセスメモリの上限 (MB)。超えたら停止
GPU = 0                     # 使用する GPU の番号
SEED = 0                    # 乱数シード（画像シャッフル・モデル初期化の再現性）
RESIZE = 256                # 画像のリサイズ後のサイズ (px)
IMAGESIZE = 224             # クロップ後の入力サイズ (px)
BACKBONE = "wideresnet50"   # 特徴抽出に使う事前学習済みバックボーン名
LAYERS = ["layer2", "layer3"]  # パッチ特徴量を取り出す中間層
PRETRAIN_DIM = 1024         # バックボーン直後の特徴次元数（前処理後）
TARGET_DIM = 1024           # メモリバンクに格納する最終的な特徴次元数
PATCHSIZE = 3               # パッチサイズ（近傍パッチを集約するカーネルサイズ）
N_NN = 1                    # 異常スコア計算に使う最近傍数
BATCH_SIZE = 8              # 特徴抽出・推論時のバッチサイズ
NUM_WORKERS = 4             # DataLoader の並列ワーカー数
FAISS_ON_GPU = True         # FAISS インデックスを GPU 上に構築するか
# --------------------------


def get_memory_mb() -> float:
    """プロセスの RSS メモリ使用量 (MB) を返す。/proc/self/status から読む。"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except OSError:
        pass
    return 0.0


def save_plot(results: list, plot_path: str, n_per_round: int, percentage: float) -> None:
    rounds = [r["round"] for r in results]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Continual Learning — PatchCore  N={n_per_round}, P={percentage}")

    ax = axes[0, 0]
    auroc_vals = [r["instance_auroc"] for r in results]
    ax.plot(rounds, auroc_vals, "b-o")
    ax.set_title("Instance AUROC")
    ax.set_xlabel("Round")
    ax.set_ylabel("AUROC")
    margin = (max(auroc_vals) - min(auroc_vals)) * 0.1 or 0.01
    ax.set_ylim(min(auroc_vals) - margin, max(auroc_vals) + margin)
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(rounds, [r["train_time"] for r in results], "r-o")
    ax.set_title("Train Time per Round (s)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Seconds")
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(rounds, [r["memory_mb"] for r in results], "g-o")
    ax.set_title("Process Memory (MB)")
    ax.set_xlabel("Round")
    ax.set_ylabel("MB")
    ax.axhline(MEMORY_LIMIT_MB, color="red", linestyle="--", linewidth=0.8, label="limit")
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(rounds, [r["memory_bank_size"] for r in results], "m-o")
    ax.set_title("Memory Bank Size (# vectors)")
    ax.set_xlabel("Round")
    ax.set_ylabel("# Vectors")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="PatchCore 継続学習実験")
    parser.add_argument("--n_per_round", type=int, default=2000,
                        help="1ラウンドあたりの学習画像枚数 (default: 2000)")
    parser.add_argument("--percentage", type=float, default=0.10,
                        help="コアセットサンプリング率 (default: 0.10)")
    parser.add_argument("--n_rounds", type=int, default=10,
                        help="継続学習のラウンド数 (default: 10)")
    args = parser.parse_args()

    N_PER_ROUND = args.n_per_round
    PERCENTAGE = args.percentage
    N_ROUNDS = args.n_rounds

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_PATH, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 学習データ全体をロード ---
    full_train = mvtec.MVTecDataset(
        DATASET_PATH,
        classname="anomaly_dataset",
        resize=RESIZE,
        imagesize=IMAGESIZE,
        split=mvtec.DatasetSplit.TRAIN,
        seed=SEED,
    )
    all_indices = np.arange(len(full_train))
    rng.shuffle(all_indices)

    needed = N_ROUNDS * N_PER_ROUND
    if len(all_indices) < needed:
        raise ValueError(
            f"学習画像が不足: {len(all_indices)} 枚 < 必要 {needed} 枚"
        )
    print(f"学習画像: {len(all_indices)} 枚  (使用: {needed} 枚)")

    # --- テストセット（固定） ---
    test_dataset = mvtec.MVTecDataset(
        DATASET_PATH,
        classname="anomaly_dataset",
        resize=RESIZE,
        imagesize=IMAGESIZE,
        split=mvtec.DatasetSplit.TEST,
        seed=SEED,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    print(f"テスト画像: {len(test_dataset)} 枚")

    # --- PatchCore 初期化 ---
    backbone = patchcore.backbones.load(BACKBONE)
    backbone.name = BACKBONE
    backbone.seed = None

    nn_method = patchcore.common.FaissNN(FAISS_ON_GPU, NUM_WORKERS)

    pc = patchcore.patchcore.PatchCore(device)
    pc.load(
        backbone=backbone,
        layers_to_extract_from=LAYERS,
        device=device,
        input_shape=(3, IMAGESIZE, IMAGESIZE),
        pretrain_embed_dimension=PRETRAIN_DIM,
        target_embed_dimension=TARGET_DIM,
        patchsize=PATCHSIZE,
        featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(
            PERCENTAGE, device
        ),
        anomaly_scorer_num_nn=N_NN,
        nn_method=nn_method,
    )

    results = []
    csv_path = os.path.join(run_dir, "results.csv")
    plot_path = os.path.join(run_dir, "progress.png")

    with open(csv_path, "w") as f:
        f.write("round,instance_auroc,train_time,memory_mb,memory_bank_size\n")

    print(f"\n{N_ROUNDS} ラウンド開始  N={N_PER_ROUND}  P={PERCENTAGE}")
    print(f"メモリ上限: {MEMORY_LIMIT_MB} MB")
    print(f"結果: {RESULTS_PATH}\n")

    for round_idx in range(N_ROUNDS):
        print(f"{'='*52}")
        print(f"  Round {round_idx + 1} / {N_ROUNDS}")
        print(f"{'='*52}")

        # 開始前メモリチェック
        mem_before = get_memory_mb()
        print(f"  メモリ (開始前): {mem_before:.0f} MB")
        if mem_before > MEMORY_LIMIT_MB:
            print(f"  !! メモリ上限 ({MEMORY_LIMIT_MB} MB) 超過 → 停止")
            break

        # このラウンドの学習画像インデックス
        start = round_idx * N_PER_ROUND
        round_indices = all_indices[start : start + N_PER_ROUND]
        round_loader = DataLoader(
            Subset(full_train, round_indices.tolist()),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        t0 = time.time()
        try:
            # パッチ特徴量の抽出
            pc.forward_modules.eval()
            new_features_list = []
            with torch.no_grad():
                for batch in round_loader:
                    imgs = batch["image"].to(torch.float).to(device)
                    new_features_list.append(pc._embed(imgs))
            new_features = np.concatenate(new_features_list, axis=0).astype(np.float32)

            # コアセットサンプリング
            new_coreset = pc.featuresampler.run(new_features).astype(np.float32)
            print(f"  新規コアセット: {len(new_coreset)} ベクトル")

            # メモリバンク更新
            #   A: 二重保持回避 — detection_features (numpy) は保持しない
            #   B: 増分 faiss add — 旧バンクを取り出して concat → fit する代わりに
            #      faiss index へ直接追加する（merge を作らない）
            if round_idx == 0:
                # 初回のみ fit() で index を構築（dim 確定 & 内部初期化）
                pc.anomaly_scorer.fit(detection_features=[new_coreset])
                pc.anomaly_scorer.detection_features = None  # A: numpy 側を即解放
            else:
                pc.anomaly_scorer.nn_method.search_index.add(new_coreset)
            bank_total = pc.anomaly_scorer.nn_method.search_index.ntotal
            print(f"  バンク合計: {bank_total} ベクトル")

        except MemoryError:
            print(f"  !! MemoryError → Round {round_idx + 1} で停止")
            break

        train_time = time.time() - t0

        # テストセットで評価
        scores, _segs, labels_gt, _gt_masks = pc.predict(test_loader)
        metrics_result = patchcore.metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt
        )
        instance_auroc = metrics_result["auroc"]

        mem_after = get_memory_mb()
        bank_size = pc.anomaly_scorer.nn_method.search_index.ntotal

        result = {
            "round": round_idx + 1,
            "instance_auroc": instance_auroc,
            "train_time": train_time,
            "memory_mb": mem_after,
            "memory_bank_size": bank_size,
        }
        results.append(result)

        print(
            f"  AUROC={instance_auroc:.4f}  "
            f"train_time={train_time:.1f}s  "
            f"memory={mem_after:.0f}MB"
        )

        # チェックポイント保存（faiss index のみ。detection_features は廃止）
        ckpt_dir = os.path.join(run_dir, f"round_{round_idx+1:02d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        pc.save_to_path(ckpt_dir)
        print(f"  [ckpt] {ckpt_dir}")

        # CSV 追記
        with open(csv_path, "a") as f:
            f.write(
                f"{result['round']},{instance_auroc:.6f},"
                f"{train_time:.2f},{mem_after:.1f},{bank_size}\n"
            )

        # グラフ更新（即時保存）
        save_plot(results, plot_path, N_PER_ROUND, PERCENTAGE)

        # ラウンド毎の中間配列を解放（メモリ累積防止）
        del new_features_list, new_features, new_coreset
        del scores, _segs, labels_gt, _gt_masks, metrics_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n完了  実行フォルダ: {run_dir}")
    print(f"      CSV: {csv_path}  グラフ: {plot_path}")


if __name__ == "__main__":
    main()
