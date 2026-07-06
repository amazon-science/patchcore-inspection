"""蒸留パイプラインを構成する各ネットワークの効率を「段」単位で計測する。

比較条件 T0(Teacher) / P1,P2(image_to_map) / P3,P4(feature_to_map) を、
  - 特徴生成ネット（WideResNet50 embed / ResNet18 backbone）
  - 異常生成ネット（faiss NN 探索 / FPN+head / FeatureToMap decoder）
の 2 段に分解し、それぞれについて
  - パラメータ数・モデルサイズ(MB)
  - 推論ピーク VRAM(MB)  ※faiss は torch 外メモリのため計測不可、bank index サイズを計上
  - 推論レイテンシ(ms/枚, batch=1) と fps
を計測して CSV と棒グラフに出力する（設計書 #5/#6）。

出力先は results/ 配下のタイムスタンプ付きフォルダ。meta.json / log.txt を同梱する。
"""
import argparse
import glob
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import patchcore.common
import patchcore.patchcore
import patchcore.runlog as runlog
from patchcore.distill_map import feature_to_map, student_fpn

plt.rcParams["font.family"] = "Noto Sans CJK JP"


def parse_args():
    p = argparse.ArgumentParser(description="ネットワーク単位の効率ベンチマーク")
    p.add_argument("--teacher", required=True, help="Teacher PatchCore モデルの保存フォルダ")
    p.add_argument("--p1", help="P1 image_to_map (正常のみ) best.ckpt")
    p.add_argument("--p2", help="P2 image_to_map (正常+疑似) best.ckpt")
    p.add_argument("--p3", help="P3 feature_to_map (正常のみ) best.ckpt")
    p.add_argument("--p4", help="P4 feature_to_map (正常+疑似) best.ckpt")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--imagesize", type=int, default=224)
    p.add_argument("--n", type=int, default=50, help="計測反復回数")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_faiss_on_gpu", action="store_true")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map"))
    p.add_argument("--name", default="bench_networks")
    return p.parse_args()


def _param_stats(module):
    n = sum(p.numel() for p in module.parameters())
    mb = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024 ** 2)
    return int(n), float(mb)


@torch.no_grad()
def _bench_fn(fn, device, n, warmup):
    """任意の呼び出し fn() を計測し (latency_ms, peak_vram_mb) を返す。"""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = (time.perf_counter() - t0) / n
    peak = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda" else None)
    return dt * 1000.0, peak


def _row(condition, stage, network, params, size_mb, vram, lat_ms, note=""):
    return {
        "condition": condition,
        "stage": stage,
        "network": network,
        "parameters": params,
        "model_size_mb": size_mb,
        "peak_vram_mb": vram,
        "latency_ms": lat_ms,
        "fps": (1000.0 / lat_ms) if lat_ms else None,
        "note": note,
    }


def _bench_teacher(pc, device, args):
    """Teacher を 特徴生成(WideResNet50 embed) と 異常生成(faiss) に分解して計測。

    戻り値: (feat_row, anom_row, feat_grid_input)
    feat_grid_input は feature_to_map decoder のベンチに使う [1,C,H',W']。
    """
    x = torch.randn(1, 3, args.imagesize, args.imagesize).to(torch.float).to(device)
    pc.forward_modules.eval()

    # 特徴生成: ResNet50 backbone + preprocessing + preadapt_aggregator
    feat_lat, feat_vram = _bench_fn(lambda: pc._embed(x), device, args.n, args.warmup)
    feat_params, feat_size = _param_stats(pc.forward_modules)

    # 異常生成用に特徴を1回だけ計算し grid 形状を取得
    features, patch_shapes = pc._embed(x, provide_patch_shapes=True)
    features = np.asarray(features)
    c = features.shape[-1]
    hh, ww = int(patch_shapes[0][0]), int(patch_shapes[0][1])
    feat_grid = torch.randn(1, c, hh, ww)

    # 異常生成: faiss NN 探索（torch 外メモリのため peak VRAM は計測不可）
    anom_lat, _ = _bench_fn(
        lambda: pc.anomaly_scorer.predict([features]), device, args.n, args.warmup
    )
    bank = pc.anomaly_scorer.nn_method.search_index
    n_bank = int(getattr(bank, "ntotal", 0) or 0)
    dim = int(getattr(bank, "d", c) or c)
    bank_mb = n_bank * dim * 4 / (1024 ** 2)  # float32 index

    feat_row = _row("T0", "特徴生成", "WideResNet50 embed", feat_params, feat_size,
                    feat_vram, feat_lat)
    anom_row = _row("T0", "異常生成", "faiss NN 探索", 0, bank_mb, None, anom_lat,
                    note=f"bank={n_bank}vec×{dim}dim, VRAMはtorch外のため未計測")
    return feat_row, anom_row, feat_grid


def _load_student(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    pattern = ckpt.get("pattern", "image_to_map")
    module = feature_to_map if pattern == "feature_to_map" else student_fpn
    student = module.build_student(**ckpt["student_kwargs"])
    student.load_state_dict(ckpt["student_state_dict"])
    student.to(device).eval()
    return student, pattern


def _bench_image_student(cond, ckpt, device, args):
    """image_to_map Student を backbone(特徴生成) と fpn+head(異常生成) に分解。"""
    student, _ = _load_student(ckpt, device)
    x = torch.randn(1, 3, args.imagesize, args.imagesize).to(device)

    feat_lat, feat_vram = _bench_fn(lambda: student.backbone(x), device, args.n, args.warmup)
    feat_params, feat_size = _param_stats(student.backbone)

    feats = student.backbone(x)  # 異常生成の入力を1回だけ計算
    anom_lat, anom_vram = _bench_fn(
        lambda: student.head(student.fpn(feats)[0]), device, args.n, args.warmup
    )
    anom_params = sum(_param_stats(m)[0] for m in (student.fpn, student.head))
    anom_size = sum(_param_stats(m)[1] for m in (student.fpn, student.head))

    return [
        _row(cond, "特徴生成", "ResNet18 backbone", feat_params, feat_size, feat_vram, feat_lat),
        _row(cond, "異常生成", "FPN+head", anom_params, anom_size, anom_vram, anom_lat),
    ]


def _bench_feature_student(cond, ckpt, device, args, feat_grid, teacher_feat_row):
    """feature_to_map Student。特徴生成は Teacher の WideResNet50 embed を共有。"""
    student, _ = _load_student(ckpt, device)
    fg = feat_grid.to(device)
    anom_lat, anom_vram = _bench_fn(lambda: student(fg), device, args.n, args.warmup)
    anom_params, anom_size = _param_stats(student)

    shared = dict(teacher_feat_row)
    shared.update(condition=cond, network="WideResNet50 embed (Teacher共有)",
                  note="特徴生成はTeacherと同一backbone")
    return [
        shared,
        _row(cond, "異常生成", "FeatureToMap decoder", anom_params, anom_size,
             anom_vram, anom_lat),
    ]


def _save_csv(rows, path):
    import csv
    fields = ["condition", "stage", "network", "parameters", "model_size_mb",
              "peak_vram_mb", "latency_ms", "fps", "note"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot(rows, metric, ylabel, title, path, logy=False):
    labels = [f"{r['condition']}\n{r['network']}" for r in rows]
    vals = [r[metric] if r[metric] is not None else 0.0 for r in rows]
    colors = ["#1f77b4" if r["stage"] == "特徴生成" else "#d62728" for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.1), 5))
    bars = ax.bar(range(len(rows)), vals, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    for b, r in zip(bars, rows):
        if r[metric] is None:
            ax.text(b.get_x() + b.get_width() / 2, 0, "N/A", ha="center",
                    va="bottom", fontsize=7, color="gray")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f77b4", label="特徴生成"),
                       Patch(color="#d62728", label="異常生成")], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    run_dir = runlog.make_run_dir(args.out, timestamp=f"{args.name}_{runlog.jst_timestamp()}")
    runlog.start_tee(run_dir)  # 復元は atexit 登録済み
    if True:
        print(f"[bench] device={device} out={run_dir}")
        print(f"[bench] teacher={args.teacher}")

        nn_method = patchcore.common.FaissNN(not args.no_faiss_on_gpu, args.num_workers)
        pc = patchcore.patchcore.PatchCore(device)
        pc.load_from_path(args.teacher, device=device, nn_method=nn_method)

        rows = []
        feat_row, anom_row, feat_grid = _bench_teacher(pc, device, args)
        rows += [feat_row, anom_row]

        image_ckpts = [("P1", args.p1), ("P2", args.p2)]
        for cond, ck in image_ckpts:
            if ck:
                print(f"[bench] {cond} image_to_map: {ck}")
                rows += _bench_image_student(cond, ck, device, args)

        feature_ckpts = [("P3", args.p3), ("P4", args.p4)]
        for cond, ck in feature_ckpts:
            if ck:
                print(f"[bench] {cond} feature_to_map: {ck}")
                rows += _bench_feature_student(cond, ck, device, args, feat_grid, feat_row)

        _save_csv(rows, os.path.join(run_dir, "networks.csv"))
        _plot(rows, "model_size_mb", "モデルサイズ (MB)", "ネットワーク別 モデルサイズ",
              os.path.join(run_dir, "bench_model_size.png"), logy=True)
        _plot(rows, "peak_vram_mb", "推論ピークVRAM (MB)", "ネットワーク別 推論ピークVRAM",
              os.path.join(run_dir, "bench_peak_vram.png"))
        _plot(rows, "latency_ms", "レイテンシ (ms/枚)", "ネットワーク別 推論レイテンシ (batch=1)",
              os.path.join(run_dir, "bench_latency.png"), logy=True)

        print("\n[bench] === 結果 ===")
        for r in rows:
            vram = "N/A" if r["peak_vram_mb"] is None else f"{r['peak_vram_mb']:.1f}MB"
            print(f"  {r['condition']:>3} {r['stage']:>6} {r['network']:<26} "
                  f"size={r['model_size_mb']:.2f}MB vram={vram} "
                  f"lat={r['latency_ms']:.3f}ms")

        runlog.write_meta(run_dir, extra={
            "teacher": args.teacher,
            "checkpoints": {"P1": args.p1, "P2": args.p2, "P3": args.p3, "P4": args.p4},
            "imagesize": args.imagesize, "n": args.n, "warmup": args.warmup,
            "gpu": args.gpu,
        })
        print(f"\n[bench] 出力完了: {run_dir}")


if __name__ == "__main__":
    main()
