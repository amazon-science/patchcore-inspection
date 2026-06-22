"""Pattern B Student（画像→異常マップ）の蒸留学習エントリポイント。

precompute 済み Teacher target（``scripts/precompute_teacher_maps.py`` 出力）を
教師信号に Student を学習する。設計書 §18.2 に対応。出力は CLAUDE.md 規約に従い
``results/distill_map/<name>_<timestamp>/`` に隔離する。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import patchcore.runlog as runlog
from patchcore.datasets import mvtec
from patchcore.distill_map import io_utils, student_fpn, trainer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Pattern B Student の蒸留学習")
    p.add_argument("targets_dir", help="precompute 出力フォルダ（train/ を含む）")
    p.add_argument("--name", default="S2_image_to_map", help="実験名（出力フォルダ接頭辞）")
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
    p.add_argument("--output_activation", default="none", choices=["none", "sigmoid"])
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
    p.add_argument("--val_ratio", type=float, default=0.05)
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
    dataset = io_utils.TeacherTargetDataset(base, train_targets_dir)
    print(f"学習サンプル: {len(dataset)} 枚（targets={train_targets_dir}）")

    score_stats = io_utils.fit_score_stats(dataset)
    normalizer = io_utils.TargetNormalizer(args.normalize, score_stats)
    print(f"normalize={args.normalize}  stats={score_stats}")

    student_kwargs = dict(
        pretrained=args.pretrained, freeze_backbone=args.freeze_backbone,
        fpn_out_channels=args.fpn_out_channels, decoder_channels=args.decoder_channels,
        target_map_size=args.imagesize, output_activation=args.output_activation,
        score_pooling=args.score_pooling, topk_ratio=args.topk_ratio,
    )
    args.student_kwargs = student_kwargs
    student = student_fpn.build_student(**student_kwargs)
    print(f"Student params(M): {student_fpn.count_parameters(student)/1e6:.3f}")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items()}, f, ensure_ascii=False, indent=2)

    result = trainer.train_student(student, dataset, normalizer, device, args, run_dir)

    runlog.write_meta(run_dir, extra={
        "name": args.name, "targets_dir": args.targets_dir, "seed": args.seed,
        "student_kwargs": student_kwargs, "normalize": args.normalize,
        "score_stats": score_stats, "best_metric": result["best_metric"],
        "epochs_run": len(result["history"]),
    })
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
