"""条件横断の異常マップ比較グリッドを作る（入力/Teacher/P1–P4/③A/③B）。

行＝test サンプル、列＝各条件の異常マップ A_S。各マップは個別に [0,1] 正規化して
jet で描画し、絶対スケールに依らず「異常の局在」を見比べられるようにする。
③A=前段→後段(凍結) / ③B=前段→後段(再学習) を P1–P4 の右に並べる。
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import patchcore.runlog as runlog
from patchcore.datasets import mvtec
from patchcore.distill_map import feature_to_map, image_to_feature, io_utils, student_fpn

plt.rcParams["font.family"] = "Noto Sans CJK JP"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_MODULE_BY_PATTERN = {"feature_to_map": feature_to_map, "image_to_feature": image_to_feature}


def parse_args():
    p = argparse.ArgumentParser(description="条件横断の異常マップ比較グリッド")
    p.add_argument("targets_dir", help="precompute 出力（test/ を含む）")
    p.add_argument("--p1", required=True, help="P1 image_to_map best.ckpt")
    p.add_argument("--p2", required=True, help="P2 image_to_map best.ckpt")
    p.add_argument("--p3", required=True, help="P3 feature_to_map best.ckpt")
    p.add_argument("--p4", required=True, help="P4 feature_to_map best.ckpt（③Aの後段にも使用）")
    p.add_argument("--front", required=True, help="③ 前段 image_to_feature best.ckpt")
    p.add_argument("--p5b_back", required=True, help="③B 後段 feature_to_map best.ckpt")
    p.add_argument("--dataset_path", default=os.path.join(REPO_ROOT, "dataset"))
    p.add_argument("--classname", default="anomaly_dataset")
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--imagesize", type=int, default=224)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num", type=int, default=8, help="先頭から並べる異常サンプル数")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map"))
    p.add_argument("--name", default="compare_conditions3")
    return p.parse_args()


def _load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pattern = ckpt.get("pattern", "image_to_map")
    module = _MODULE_BY_PATTERN.get(pattern, student_fpn)
    m = module.build_student(**ckpt["student_kwargs"])
    m.load_state_dict(ckpt["student_state_dict"])
    m.to(device).eval()
    return m, pattern


class _Composed(torch.nn.Module):
    def __init__(self, front, back):
        super().__init__()
        self.front, self.back = front, back

    def forward(self, x):
        return self.back(self.front(x))


@torch.no_grad()
def _amap(student, pattern, image, feat, device):
    """条件ごとに A_S [H,W] を返す。pattern で入力（画像 or 特徴）を切替。"""
    if pattern == "feature_to_map":
        a = student(feat.unsqueeze(0).to(device))
    else:  # image_to_map / composed(③)
        a = student(image.unsqueeze(0).to(device))
    return a[0, 0].detach().cpu().numpy()


def _norm01(m):
    mn, mx = float(m.min()), float(m.max())
    return (m - mn) / (mx - mn + 1e-8)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    run_dir = runlog.make_run_dir(args.out, timestamp=f"{args.name}_{runlog.jst_timestamp()}")
    runlog.start_tee(run_dir)
    print(f"出力先: {run_dir}\nDevice: {device}")

    base = mvtec.MVTecDataset(
        args.dataset_path, classname=args.classname, resize=args.resize,
        imagesize=args.imagesize, split=mvtec.DatasetSplit.TEST, seed=42,
    )
    ds = io_utils.TeacherTargetDataset(
        base, os.path.join(args.targets_dir, "test"), load_feature=True)

    # 異常サンプルを sample_id 昇順で num 枚選ぶ
    picked = []
    for j in range(len(ds)):
        sid = ds.items[j][1]
        if "abnormal" in sid:
            picked.append(j)
    picked = sorted(picked, key=lambda j: ds.items[j][1])[: args.num]
    print(f"選択サンプル: {[ds.items[j][1].split('__')[-1] for j in picked]}")

    # 条件（表示名, checkpoint or 合成, pattern）
    p1, _ = _load(args.p1, device)
    p2, _ = _load(args.p2, device)
    p3, _ = _load(args.p3, device)
    p4, _ = _load(args.p4, device)
    front, _ = _load(args.front, device)
    p5b, _ = _load(args.p5b_back, device)
    a3 = _Composed(front, p4).to(device).eval()   # ③A: 前段→後段(凍結=P4)
    b3 = _Composed(front, p5b).to(device).eval()   # ③B: 前段→後段(再学習)

    conditions = [
        ("P1 画像/正常", p1, "image_to_map"),
        ("P2 画像/疑似", p2, "image_to_map"),
        ("P3 特徴/正常", p3, "feature_to_map"),
        ("P4 特徴/疑似", p4, "feature_to_map"),
        ("③A 前段→凍結", a3, "image_to_map"),
        ("③B 前段→再学習", b3, "image_to_map"),
    ]
    col_titles = ["入力画像", "Teacher A_T"] + [c[0] for c in conditions]
    ncol = len(col_titles)
    nrow = len(picked)

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.1, nrow * 2.1))
    if nrow == 1:
        axes = axes[None, :]

    for r, j in enumerate(picked):
        item = ds[j]
        image = item["image"]
        feat = item["teacher_feature"]
        a_t = item["teacher_map"][0].numpy()  # [H,W] 生
        sid = ds.items[j][1].split("__")[-1].replace(".jpg", "")

        # 入力画像（ImageNet 正規化を戻して表示）
        img = image.numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        axes[r, 0].imshow(np.clip(img, 0, 1))
        axes[r, 0].set_ylabel(sid, fontsize=9)

        axes[r, 1].imshow(_norm01(a_t), cmap="jet", vmin=0, vmax=1)

        for c, (_, student, pattern) in enumerate(conditions):
            a_s = _amap(student, pattern, image, feat, device)
            axes[r, 2 + c].imshow(_norm01(a_s), cmap="jet", vmin=0, vmax=1)

        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=11)

    fig.suptitle("異常マップ 条件比較（各マップ [0,1] 正規化）", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_png = os.path.join(run_dir, f"compare_{len(conditions)}conditions.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"保存: {out_png}")

    runlog.write_meta(run_dir, extra={
        "targets_dir": args.targets_dir, "num": args.num,
        "checkpoints": {"p1": args.p1, "p2": args.p2, "p3": args.p3, "p4": args.p4,
                        "front": args.front, "p5b_back": args.p5b_back},
    })
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
