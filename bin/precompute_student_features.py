"""蒸留方法③-B 用: 凍結した前段(image_to_feature)の出力 F_S を事前計算して保存する。

前段 P5 は凍結・学習時のデータ拡張も無い（決定的変換）ため、各画像に対する F_S は
エポック不変。よって一度だけ前段を通して F_S を計算・保存し、後段の学習は既存の
高速な feature_to_map 経路（backbone forward 不要）で回せる。これにより ③-B の
後段再学習を「画像→前段→後段を毎エポック」から「F_S→後段のみ」に置き換えて高速化する。

出力は既存 precompute と同一フォーマットの .pt（`teacher_feature_map` を F_S に差し替え、
`teacher_anomaly_map`/`teacher_anomaly_score`/`label`/`image_path` は元の Teacher 値を継承）。
生成先を train_distill_map.py に `--pattern feature_to_map` で渡せば後段が F_S→A_T を学習する。
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import patchcore.runlog as runlog
from patchcore.datasets import mvtec
from patchcore.distill_map import image_to_feature, io_utils

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="凍結前段の出力 F_S を事前計算（③-B 高速化用）")
    p.add_argument("--front_checkpoint", required=True, help="前段 image_to_feature の best.ckpt")
    p.add_argument("--targets_dir", required=True,
                   help="元 precompute フォルダ（train/ を含む。A_T等を継承）")
    p.add_argument("--pseudo_dir", default=None, help="元 疑似 precompute の split フォルダ（任意）")
    p.add_argument("--dataset_path", default=os.path.join(REPO_ROOT, "dataset"))
    p.add_argument("--classname", default="anomaly_dataset")
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--imagesize", type=int, default=224)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_per_split", type=int, default=None, help="スモーク用の枚数上限")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map",
                                                 "precomputed_student_feat"))
    p.add_argument("--name", default="P5_FS")
    return p.parse_args()


def _load_front(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt.get("pattern") != "image_to_feature":
        raise ValueError(f"--front_checkpoint は image_to_feature が必要: pattern={ckpt.get('pattern')}")
    front = image_to_feature.build_student(**ckpt["student_kwargs"])
    front.load_state_dict(ckpt["student_state_dict"])
    front.to(device).eval()
    return front


@torch.no_grad()
def _process(front, dataset, device, split_dir, args):
    os.makedirs(split_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    limit = args.max_per_split or len(dataset)
    done = 0
    for batch in loader:
        if done >= limit:
            break
        images = batch["image"].to(device, non_blocking=True)
        f_s = front(images).cpu()  # [B,C,H',W']
        amaps = batch["teacher_map"]      # [B,1,H,W] float（生 A_T）
        scores = batch["teacher_score"]
        labels = batch["label"]
        sids = batch["sample_id"]
        ipaths = batch["image_path"]
        for i in range(images.shape[0]):
            if done >= limit:
                break
            amap = amaps[i]
            if amap.dim() == 3:
                amap = amap.squeeze(0)  # [H,W]（元 precompute と同形式）
            record = {
                "image_path": ipaths[i],
                "teacher_anomaly_map": amap.clone(),
                "teacher_anomaly_score": float(scores[i]),
                "label": int(labels[i]),
                "teacher_feature_map": f_s[i].half(),  # ← F_S を格納
            }
            torch.save(record, os.path.join(split_dir, sids[i] + ".pt"))
            done += 1
        print(f"  [{os.path.basename(split_dir)}] {done}/{limit} 枚保存", flush=True)
    return done


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    run_dir = runlog.make_run_dir(args.out, timestamp=f"{args.name}_{runlog.jst_timestamp()}")
    runlog.start_tee(run_dir)
    print(f"出力先: {run_dir}\nDevice: {device}")
    print(f"前段: {args.front_checkpoint}")

    front = _load_front(args.front_checkpoint, device)

    # 正常: mvtec base 経由で画像を正しく解決（image_path の相対不整合を回避）
    base = mvtec.MVTecDataset(
        args.dataset_path, classname=args.classname, resize=args.resize,
        imagesize=args.imagesize, split=mvtec.DatasetSplit.TRAIN, seed=0,
    )
    normal = io_utils.TeacherTargetDataset(
        base, os.path.join(args.targets_dir, "train"), load_feature=False,
    )
    n_normal = _process(front, normal, device, os.path.join(run_dir, "train"), args)
    print(f"正常 F_S 保存: {n_normal} 枚")

    n_pseudo = 0
    if args.pseudo_dir:
        pseudo = io_utils.DirectImageTargetDataset(
            args.pseudo_dir, resize=args.resize, imagesize=args.imagesize, load_feature=False,
        )
        n_pseudo = _process(front, pseudo, device, os.path.join(run_dir, "pseudo"), args)
        print(f"疑似 F_S 保存: {n_pseudo} 枚")

    runlog.write_meta(run_dir, extra={
        "front_checkpoint": args.front_checkpoint, "targets_dir": args.targets_dir,
        "pseudo_dir": args.pseudo_dir, "n_normal": n_normal, "n_pseudo": n_pseudo,
    })
    print(f"完了: {run_dir}")
    print(f"→ 後段学習: bin/train_distill_map.py {run_dir} "
          f"{'--pseudo_dir ' + os.path.join(run_dir, 'pseudo') if n_pseudo else ''} "
          f"--pattern feature_to_map")


if __name__ == "__main__":
    main()
