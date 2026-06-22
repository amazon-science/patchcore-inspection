"""Pattern B Student の評価エントリポイント（設計書 §18.3）。

学習済み Student を test set で推論し、(1) image-level metrics、(2) 蒸留忠実度、
(3) 効率指標、(4) 可視化、(5) scores.csv / json を出力する。``--teacher`` を渡すと
Teacher PatchCore の効率（memory bank search 込み）も同条件で計測し、T0 vs S2 を比較する。
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import patchcore.metrics as pmetrics
import patchcore.runlog as runlog
from patchcore.datasets import mvtec
from patchcore.distill_map import io_utils, student_fpn, visualize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Pattern B Student の評価")
    p.add_argument("targets_dir", help="precompute 出力フォルダ（test/ を含む）")
    p.add_argument("--checkpoint", required=True, help="Student の best.ckpt")
    p.add_argument("--name", default="eval_S2", help="出力フォルダ接頭辞")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_path", default=os.path.join(REPO_ROOT, "dataset"))
    p.add_argument("--classname", default="anomaly_dataset")
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--imagesize", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_test", type=int, default=None)
    p.add_argument("--num_vis", type=int, default=8, help="可視化するサンプル数")
    p.add_argument("--teacher", default=None, help="(任意) Teacher 効率計測用フォルダ")
    p.add_argument("--no_faiss_on_gpu", action="store_true")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map"))
    return p.parse_args()


def _anomaly_type(image_path: str) -> str:
    parts = image_path.split("/")
    return parts[-2] if len(parts) >= 2 else "unknown"


def _load_student(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    student = student_fpn.build_student(**ckpt["student_kwargs"])
    student.load_state_dict(ckpt["student_state_dict"])
    student.to(device).eval()
    normalizer = io_utils.TargetNormalizer(
        ckpt["normalizer"]["mode"], ckpt["normalizer"]["stats"]
    )
    return student, normalizer


@torch.no_grad()
def _student_efficiency(student, device, imagesize, n=50):
    x = torch.randn(1, 3, imagesize, imagesize, device=device)
    for _ in range(5):  # warmup
        student(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(n):
        student(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    mem = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
           if device.type == "cuda" else None)
    return {
        "inference_time_ms": dt * 1000,
        "fps": 1.0 / dt,
        "parameters": int(student_fpn.count_parameters(student)),
        "model_size_mb": student_fpn.model_size_mb(student),
        "gpu_memory_mb": mem,
    }


def _teacher_efficiency(teacher_dir, device, args, n_images):
    import patchcore.common
    import patchcore.patchcore
    nn_method = patchcore.common.FaissNN(not args.no_faiss_on_gpu, args.num_workers)
    pc = patchcore.patchcore.PatchCore(device)
    pc.load_from_path(teacher_dir, device=device, nn_method=nn_method)
    x = torch.randn(1, 3, args.imagesize, args.imagesize)
    for _ in range(3):
        pc._predict(x)  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = max(10, min(n_images, 50))
    for _ in range(reps):
        pc._predict(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    bank = pc.anomaly_scorer.nn_method.search_index
    n_bank = getattr(bank, "ntotal", None)
    return {
        "inference_time_ms": dt * 1000,
        "fps": 1.0 / dt,
        "bank_vectors": int(n_bank) if n_bank else None,
        "note": "memory bank search 込み（バッチ1）",
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    run_dir = runlog.make_run_dir(args.out, timestamp=f"{args.name}_{runlog.jst_timestamp()}")
    runlog.start_tee(run_dir)
    print(f"出力先: {run_dir}\nDevice: {device}")

    student, normalizer = _load_student(args.checkpoint, device)
    print(f"Student ロード: {args.checkpoint}  normalize={normalizer.mode}")

    base = mvtec.MVTecDataset(
        args.dataset_path, classname=args.classname, resize=args.resize,
        imagesize=args.imagesize, split=mvtec.DatasetSplit.TEST, seed=args.seed,
    )
    dataset = io_utils.TeacherTargetDataset(base, os.path.join(args.targets_dir, "test"))
    if args.max_test:
        dataset = Subset(dataset, list(range(min(args.max_test, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    print(f"テスト: {len(dataset)} 枚")

    rows = []
    map_se_sum = map_ae_sum = px_count = 0.0
    vis_dir = os.path.join(run_dir, "visualizations")
    n_vis_done = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            a_t_raw = batch["teacher_map"].to(device)          # [B,1,H,W]
            a_t = normalizer.apply(a_t_raw)
            a_s = student(images)
            s_s = student.score(a_s)
            for i in range(images.shape[0]):
                se = float(((a_s[i] - a_t[i]) ** 2).mean())
                ae = float((a_s[i] - a_t[i]).abs().mean())
                npx = a_s[i].numel()
                map_se_sum += se * npx
                map_ae_sum += ae * npx
                px_count += npx
                ipath = batch["image_path"][i]
                rows.append({
                    "sample_id": batch["sample_id"][i],
                    "image_path": ipath,
                    "label": int(batch["label"][i]),
                    "anomaly_type": _anomaly_type(ipath),
                    "teacher_score": float(batch["teacher_score"][i]),
                    "student_score": float(s_s[i]),
                    "map_mse": se,
                })
                if n_vis_done < args.num_vis:
                    visualize.save_panel(
                        vis_dir, batch["sample_id"][i], images[i].cpu(),
                        a_t_raw[i, 0].cpu(), a_s[i].cpu(),
                        score_info=(f"{batch['sample_id'][i]}  "
                                    f"s_T={float(batch['teacher_score'][i]):.3f} "
                                    f"s_S={float(s_s[i]):.3f}"),
                    )
                    n_vis_done += 1

    labels = np.array([r["label"] for r in rows])
    t_scores = np.array([r["teacher_score"] for r in rows])
    s_scores = np.array([r["student_score"] for r in rows])

    def _img_metrics(scores):
        m = pmetrics.compute_imagewise_retrieval_metrics(scores, labels)
        from sklearn.metrics import average_precision_score
        return {
            "auroc": float(m["auroc"]),
            "average_precision": float(average_precision_score(labels, scores)),
            "f1_max": float(m["f1"]),
        }

    image_metrics = {"teacher": _img_metrics(t_scores), "student": _img_metrics(s_scores)}

    from scipy.stats import pearsonr, spearmanr
    s_t_norm = normalizer.apply(torch.tensor(t_scores)).numpy()
    fidelity = {
        "map_mse": map_se_sum / px_count,
        "map_mae": map_ae_sum / px_count,
        "score_mse": float(np.mean((s_scores - s_t_norm) ** 2)),
        "score_mae": float(np.mean(np.abs(s_scores - s_t_norm))),
        "score_pearson": float(pearsonr(s_scores, t_scores)[0]),
        "score_spearman": float(spearmanr(s_scores, t_scores)[0]),
    }

    efficiency = {"student": _student_efficiency(student, device, args.imagesize)}
    if args.teacher:
        print("Teacher 効率を計測中...")
        efficiency["teacher"] = _teacher_efficiency(args.teacher, device, args, len(rows))

    # --- 保存 ---
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
    for fname, obj in (
        ("metrics/image_metrics.json", image_metrics),
        ("metrics/distillation_fidelity.json", fidelity),
        ("metrics/efficiency_metrics.json", efficiency),
    ):
        with open(os.path.join(run_dir, fname), "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "predictions", "scores.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    visualize.save_roc(
        os.path.join(run_dir, "roc_teacher_vs_student.png"), labels, t_scores, s_scores,
        {"Teacher": image_metrics["teacher"]["auroc"],
         "Student": image_metrics["student"]["auroc"]},
    )

    print("=== image metrics ===")
    print(json.dumps(image_metrics, ensure_ascii=False, indent=2))
    print("=== distillation fidelity ===")
    print(json.dumps(fidelity, ensure_ascii=False, indent=2))
    print("=== efficiency ===")
    print(json.dumps(efficiency, ensure_ascii=False, indent=2))

    runlog.write_meta(run_dir, extra={
        "name": args.name, "checkpoint": args.checkpoint, "targets_dir": args.targets_dir,
        "teacher": args.teacher, "n_test": len(rows),
        "image_metrics": image_metrics, "fidelity": fidelity, "efficiency": efficiency,
    })
    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
