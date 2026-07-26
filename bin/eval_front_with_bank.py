#!/usr/bin/env python3
"""③前段（画像→特徴map）の Student 特徴 F_S を PatchCore メモリバンクに通して評価する。

定量評価表の「画像→特徴map」行を埋めるためのスクリプト。この行は Student が異常マップを
出さないため、**後段は Teacher の faiss メモリバンクをそのまま使う**：

    画像 →[前段 ResNet18-FPN]→ F_S →[Teacher メモリバンク NN 探索]→ 異常マップ/スコア

議事録 20260707 で先生Bが挙げた「ResNet50 でメモリバンクを作り、18 でそのメモリバンクと
見比べる一番素直な構成」に対応する。PatchCore の骨子（中間特徴の一致＋メモリバンク）を
保ったまま前段だけ軽量化した場合の精度と、先生B が指摘した
「メモリバンク側の計算量が支配的で結局速くならない」を定量化する。

出力：
- 特徴忠実度 feat_MSE / feat_MAE（F_S vs Teacher F_T）＝表の「特徴mapに対する評価」列
- Instance AUROC / AP / F1max（メモリバンク推論）＝表の「異常mapに対する評価」列
- end-to-end 推論時間（前段＋NN探索）とメモリ（モデル＋bank）＝表の「パフォーマンス」列

CLAUDE.md 準拠：出力は results/ 配下のタイムスタンプ付きフォルダ、meta.json / log.txt。
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import patchcore.common
import patchcore.patchcore
import patchcore.runlog as runlog
from patchcore.datasets import mvtec
from patchcore.distill_map import image_to_feature, io_utils

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="③前段 F_S を Teacher メモリバンクに通して評価")
    p.add_argument("targets_dir", help="precompute 出力（test/ に teacher_feature を含むこと）")
    p.add_argument("--front_checkpoint", required=True, help="前段 image_to_feature の best.ckpt")
    p.add_argument("--teacher", required=True, help="Teacher PatchCore フォルダ（メモリバンク）")
    p.add_argument("--name", default="eval_front_bank")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_test", type=int, default=None, help="スモーク用の枚数上限")
    p.add_argument("--dataset_path", default=os.path.join(REPO_ROOT, "dataset"))
    p.add_argument("--classname", default="anomaly_dataset")
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--imagesize", type=int, default=224)
    p.add_argument("--no_faiss_on_gpu", action="store_true")
    p.add_argument("--bench_n", type=int, default=50, help="推論時間の計測反復数")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map"))
    return p.parse_args()


def _load_front(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt.get("pattern") != "image_to_feature":
        raise ValueError(f"--front_checkpoint は image_to_feature が必要: pattern={ckpt.get('pattern')}")
    front = image_to_feature.build_student(**ckpt["student_kwargs"])
    front.load_state_dict(ckpt["student_state_dict"])
    front.to(device).eval()
    n_params = sum(p.numel() for p in front.parameters())
    return front, n_params


@torch.no_grad()
def _bank_predict(pc, feat_grid):
    """F_S([B,C,H',W']) を Teacher メモリバンクに通して (scores, maps) を返す。

    ``PatchCore._predict`` の backbone 部分だけを F_S に差し替えた経路。
    """
    b, c, h, w = feat_grid.shape
    # [B,C,H',W'] → [B*P, C]（_embed の出力並びに合わせる）
    features = feat_grid.permute(0, 2, 3, 1).reshape(-1, c).contiguous()
    features = np.asarray(features.cpu().numpy(), dtype=np.float32)

    patch_scores = image_scores = pc.anomaly_scorer.predict([features])[0]
    image_scores = pc.patch_maker.unpatch_scores(image_scores, batchsize=b)
    image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
    image_scores = pc.patch_maker.score(image_scores)

    patch_scores = pc.patch_maker.unpatch_scores(patch_scores, batchsize=b)
    patch_scores = patch_scores.reshape(b, h, w)
    masks = pc.anomaly_segmentor.convert_to_segmentation(patch_scores)
    return list(image_scores), list(masks)


def _f1_max(y, s):
    prec, rec, _ = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(np.nanmax(f1))


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    run_dir = runlog.make_run_dir(args.out, timestamp=f"{args.name}_{runlog.jst_timestamp()}")
    runlog.start_tee(run_dir)
    print(f"出力先: {run_dir}\nDevice: {device}")

    front, n_params = _load_front(args.front_checkpoint, device)
    print(f"前段ロード: {args.front_checkpoint}  params={n_params/1e6:.2f}M")

    nn_method = patchcore.common.FaissNN(not args.no_faiss_on_gpu, args.num_workers)
    pc = patchcore.patchcore.PatchCore(device)
    pc.load_from_path(args.teacher, device=device, nn_method=nn_method)
    print(f"Teacher メモリバンク: {args.teacher}")

    # test 画像は mvtec base 経由で解決する（precompute の image_path は相対のため）
    base = mvtec.MVTecDataset(
        args.dataset_path, classname=args.classname, resize=args.resize,
        imagesize=args.imagesize, split=mvtec.DatasetSplit.TEST, seed=0,
    )
    ds = io_utils.TeacherTargetDataset(
        base, os.path.join(args.targets_dir, "test"), load_feature=True)
    if args.max_test:
        ds.items = ds.items[:args.max_test]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    print(f"test: {len(ds)} 枚")

    feat_se = feat_ae = feat_n = 0.0
    s_student, s_teacher, labels = [], [], []
    t_front = t_bank = 0.0
    n_img = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        f_t = batch["teacher_feature"].to(device)          # [B,C,H',W'] Teacher F_T
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        f_s = front(images)                                 # [B,C,H',W'] Student F_S
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_front += time.time() - t0

        d = (f_s.float() - f_t.float())
        feat_se += float((d ** 2).sum()); feat_ae += float(d.abs().sum())
        feat_n += d.numel()

        t0 = time.time()
        scores, _ = _bank_predict(pc, f_s)                  # メモリバンク NN 探索
        t_bank += time.time() - t0

        s_student.extend([float(v) for v in scores])
        s_teacher.extend([float(v) for v in batch["teacher_score"]])
        labels.extend([int(v) for v in batch["label"]])
        n_img += images.shape[0]
        print(f"  {n_img}/{len(ds)} 枚", flush=True)

    y = np.asarray(labels)
    ss = np.asarray(s_student, dtype=np.float64)
    st = np.asarray(s_teacher, dtype=np.float64)
    metrics = {
        "feature_fidelity": {"feat_mse": feat_se / feat_n, "feat_mae": feat_ae / feat_n},
        "student_via_bank": {
            "auroc": float(roc_auc_score(y, ss)),
            "average_precision": float(average_precision_score(y, ss)),
            "f1_max": _f1_max(y, ss),
        },
        "teacher": {
            "auroc": float(roc_auc_score(y, st)),
            "average_precision": float(average_precision_score(y, st)),
            "f1_max": _f1_max(y, st),
        },
        "score_agreement": {
            "pearson": float(np.corrcoef(ss, st)[0, 1]),
            "spearman": float(np.corrcoef(np.argsort(np.argsort(ss)),
                                          np.argsort(np.argsort(st)))[0, 1]),
        },
    }

    # ---- end-to-end 効率（前段 + メモリバンク NN）----
    bank_mb = float(pc.anomaly_scorer.nn_method.search_index.ntotal
                    * 1024 * 4 / 1024 / 1024) if hasattr(
        pc.anomaly_scorer.nn_method, "search_index") else None
    metrics["efficiency"] = {
        "front_ms_per_image": t_front / max(n_img, 1) * 1000,
        "bank_ms_per_image": t_bank / max(n_img, 1) * 1000,
        "end_to_end_ms_per_image": (t_front + t_bank) / max(n_img, 1) * 1000,
        "front_parameters": int(n_params),
        "front_model_size_mb": n_params * 4 / 1024 / 1024,
        "memory_bank_mb": bank_mb,
        "total_memory_mb": (n_params * 4 / 1024 / 1024) + (bank_mb or 0),
    }

    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    with open(os.path.join(run_dir, "metrics", "front_bank_metrics.json"), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    np.savetxt(os.path.join(run_dir, "scores.csv"),
               np.stack([y, ss, st], 1), delimiter=",", header="label,student_bank,teacher",
               comments="", fmt="%.6f")

    # ---- 図：スコア分布 ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, (s, tag, auc) in zip(ax, ((ss, "Student F_S → メモリバンク", metrics["student_via_bank"]["auroc"]),
                                     (st, "Teacher (F_T → メモリバンク)", metrics["teacher"]["auroc"]))):
        a.hist(s[y == 0], bins=40, alpha=0.6, label=f"正常 n={int((y==0).sum())}", density=True)
        a.hist(s[y == 1], bins=40, alpha=0.6, label=f"異常 n={int((y==1).sum())}", density=True)
        a.set_title(f"{tag}\nAUROC={auc:.4f}"); a.set_xlabel("異常スコア"); a.legend()
    ax[0].set_ylabel("確率密度")
    fig.suptitle("③前段 F_S を Teacher メモリバンクに通した評価", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "score_distribution.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    runlog.write_meta(run_dir, extra={
        "front_checkpoint": args.front_checkpoint, "teacher": args.teacher,
        "targets_dir": args.targets_dir, "n_test": int(len(ds)), "metrics": metrics,
    })
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
