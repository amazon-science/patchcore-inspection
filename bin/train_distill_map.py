"""Pattern B Student（画像→異常マップ）の蒸留学習エントリポイント。

precompute 済み Teacher target（``scripts/precompute_teacher_maps.py`` 出力）を
教師信号に Student を学習する。設計書 §18.2 に対応。出力は CLAUDE.md 規約に従い
``results/distill_map/<name>_<timestamp>/`` に隔離する。
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import patchcore.runlog as runlog
from patchcore.datasets import mvtec
from patchcore.distill_map import (
    feature_to_map,
    image_to_feature,
    io_utils,
    student_fpn,
    trainer,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Pattern B Student の蒸留学習")
    p.add_argument("targets_dir", help="precompute 出力フォルダ（train/ を含む）")
    p.add_argument("--pseudo_dir", default=None,
                   help="(S3) 疑似異常 precompute の split フォルダ（例 <run>/pseudo）。normal と結合学習")
    p.add_argument("--pseudo_max", type=int, default=0,
                   help=">0 で pseudo の先頭 N 枚のみ使用（100倍プールから 5/10/50 倍を切り出す）")
    p.add_argument("--pseudo_target_scale", type=float, default=0.0,
                   help="A(マスク直接)ターゲットの倍率。0=オート（正常スコア最大に合わせPatchCore異常"
                        "スケールへ整合）。マスク0-1のまま混ぜると異常が正常に埋もれるため必須")
    p.add_argument("--name", default="S2_image_to_map", help="実験名（出力フォルダ接頭辞）")
    p.add_argument("--pattern", default="image_to_map",
                   choices=["image_to_map", "feature_to_map", "image_to_feature"],
                   help="B=image_to_map / A=feature_to_map / ③前段=image_to_feature"
                        "（A・③前段は precompute --save_feature 必須）")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    # dataset（画像の前処理は Teacher precompute と一致させる）
    p.add_argument("--dataset_path", default=os.path.join(REPO_ROOT, "dataset"))
    p.add_argument("--classname", default="anomaly_dataset")
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--imagesize", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    # student
    p.add_argument("--pretrained", type=int, default=1)
    p.add_argument("--freeze_backbone", type=int, default=0)
    p.add_argument("--fpn_out_channels", type=int, default=128)
    p.add_argument("--decoder_channels", type=int, default=128)
    p.add_argument("--reduce_channels", type=int, default=256,
                   help="Pattern A: 1x1 縮約後のチャンネル数")
    p.add_argument("--output_activation", default="none", choices=["none", "sigmoid"])
    p.add_argument("--feat_output_activation", default="relu", choices=["none", "relu"],
                   help="③前段(image_to_feature)の出力活性化。生F_T(post-ReLU非負)に合わせ既定relu")
    p.add_argument("--score_pooling", default="max", choices=["max", "topk_mean", "mean"])
    p.add_argument("--topk_ratio", type=float, default=0.01)
    # normalization（map/score 共通アフィン）
    p.add_argument("--normalize", default="minmax", choices=list(io_utils.TargetNormalizer.MODES))
    # loss
    p.add_argument("--map_loss", default="smooth_l1", choices=["mse", "smooth_l1"])
    p.add_argument("--lambda_map", type=float, default=1.0)
    p.add_argument("--lambda_score", type=float, default=0.1)
    p.add_argument("--use_weighted_map_loss", type=int, default=1)
    p.add_argument("--anomaly_weight_alpha", type=float, default=5.0)
    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.05,
                   help="normal/pseudo それぞれの hold-out（未使用・非オーバーサンプル）割合")
    p.add_argument("--normal_pseudo_ratio", type=float, default=1.0,
                   help="バッチ内 normal:pseudo をこの比(=r:1)に保つ（1.0=1:1）。pseudo使用時のみ")
    p.add_argument("--steps_per_epoch", type=int, default=0,
                   help=">0 で1epochのステップ数を固定（正常の過反復による肥大化を抑制）")
    p.add_argument("--probe_size", type=int, default=2000,
                   help="過学習ギャップ用に評価する train 正常の枚数")
    p.add_argument("--channels_last", type=int, default=1, help="channels_last メモリ形式（CUDA）")
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--cache_normal", type=int, default=1,
                   help="正常 target を RAM 常駐させ 1:1 の正常側 IO を消す（B1）")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--early_stopping", type=int, default=1)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map"))
    args = p.parse_args()
    # int フラグを bool に
    for k in ("pretrained", "freeze_backbone", "use_weighted_map_loss",
              "amp", "early_stopping"):
        setattr(args, k, bool(getattr(args, k)))
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    run_dir = runlog.make_run_dir(args.out, timestamp=f"{args.name}_{runlog.jst_timestamp()}")
    runlog.start_tee(run_dir)
    print(f"出力先: {run_dir}\nDevice: {device}")

    base = mvtec.MVTecDataset(
        args.dataset_path, classname=args.classname, resize=args.resize,
        imagesize=args.imagesize, split=mvtec.DatasetSplit.TRAIN, seed=args.seed,
    )
    train_targets_dir = os.path.join(args.targets_dir, "train")
    load_feature = args.pattern in ("feature_to_map", "image_to_feature")
    normal_ds = io_utils.TeacherTargetDataset(base, train_targets_dir, load_feature=load_feature)
    stat_sources = [normal_ds]
    pseudo_ds = None
    if args.pseudo_dir:
        # A(マスク)ターゲットのスケール整合：正常のみ統計から倍率を決める（0=オート）
        if args.pseudo_target_scale > 0:
            pscale = args.pseudo_target_scale
        else:
            nstats = io_utils.fit_score_stats([normal_ds])
            pscale = float(nstats["max"])
        pmax = args.pseudo_max or None
        if io_utils.is_packed(args.pseudo_dir):
            pseudo_ds = io_utils.PackedTargetDataset(
                args.pseudo_dir, load_feature=load_feature, in_memory=False,
                max_count=pmax, target_scale=pscale)
            print(f"pseudo: packed memmap を使用（{len(pseudo_ds)} 枚, pseudo_max={args.pseudo_max}）")
        else:
            pseudo_ds = io_utils.DirectImageTargetDataset(
                args.pseudo_dir, resize=args.resize, imagesize=args.imagesize,
                load_feature=load_feature, max_count=pmax, target_scale=pscale,
            )
        print(f"pseudo target_scale={pscale:.3f}（マスク→PatchCore異常スケール整合）")
        stat_sources.append(pseudo_ds)
        print(f"学習サンプル: normal {len(normal_ds)} + pseudo {len(pseudo_ds)} 枚"
              f"（pattern={args.pattern}）")
    else:
        print(f"学習サンプル: normal {len(normal_ds)} 枚（targets={train_targets_dir}, pattern={args.pattern}）")

    # 正規化統計は cache 前の元 dataset（stat_sources）から算出する
    score_stats = io_utils.fit_score_stats(stat_sources)

    # 正常を RAM 常駐（1:1 の正常側 IO を消す）。feature shape 参照にも使う
    if args.cache_normal:
        normal_ds = io_utils.InMemoryCacheDataset(normal_ds)
    dataset = normal_ds
    normalizer = io_utils.TargetNormalizer(args.normalize, score_stats)
    print(f"normalize={args.normalize}  stats={score_stats}")

    common = dict(
        target_map_size=args.imagesize, output_activation=args.output_activation,
        score_pooling=args.score_pooling, topk_ratio=args.topk_ratio,
    )
    if args.pattern == "image_to_feature":
        feat = dataset[0]["teacher_feature"]  # [C,H',W']
        c_out, h_feat = int(feat.shape[0]), int(feat.shape[1])
        student_kwargs = dict(
            out_channels=c_out, target_feat_size=h_feat,
            pretrained=args.pretrained, freeze_backbone=args.freeze_backbone,
            fpn_out_channels=args.fpn_out_channels, decoder_channels=args.decoder_channels,
            output_activation=args.feat_output_activation,
        )
        args.input_key = "image"
        student = image_to_feature.build_student(**student_kwargs)
        n_params = image_to_feature.count_parameters(student)
        print(f"③前段 image→feature: out_ch={c_out} 格子={h_feat}x{h_feat} "
              f"act={args.feat_output_activation}")
    elif args.pattern == "feature_to_map":
        feat = dataset[0]["teacher_feature"]  # [C,H',W']
        c_in, h_feat = int(feat.shape[0]), int(feat.shape[1])
        num_up = max(1, round(math.log2(args.imagesize / h_feat)))
        student_kwargs = dict(
            in_channels=c_in, reduce_channels=args.reduce_channels,
            decoder_channels=args.decoder_channels, num_up=num_up, **common,
        )
        args.input_key = "teacher_feature"
        student = feature_to_map.build_student(**student_kwargs)
        n_params = feature_to_map.count_parameters(student)
        print(f"Pattern A 入力特徴: C={c_in} H'={h_feat} → num_up={num_up}")
    else:
        student_kwargs = dict(
            pretrained=args.pretrained, freeze_backbone=args.freeze_backbone,
            fpn_out_channels=args.fpn_out_channels, decoder_channels=args.decoder_channels,
            **common,
        )
        args.input_key = "image"
        student = student_fpn.build_student(**student_kwargs)
        n_params = student_fpn.count_parameters(student)
    args.student_kwargs = student_kwargs
    print(f"Student params(M): {n_params/1e6:.3f}")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items()}, f, ensure_ascii=False, indent=2)

    result = trainer.train_student(student, normal_ds, normalizer, device, args, run_dir,
                                   pseudo_ds=pseudo_ds)

    runlog.write_meta(run_dir, extra={
        "name": args.name, "targets_dir": args.targets_dir, "seed": args.seed,
        "student_kwargs": student_kwargs, "normalize": args.normalize,
        "score_stats": score_stats, "best_metric": result["best_metric"],
        "epochs_run": len(result["history"]),
    })
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
