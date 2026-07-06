"""蒸留の最終比較表を、各条件の eval 出力（metrics/*.json）から横断集計する。

T0(Teacher) / P1–P4 / ③A / ③B を1つの表・図にまとめる。各条件の eval ディレクトリ
（`eval_distill_map.py` が生成、metrics/image_metrics.json・distillation_fidelity.json を含む）
を `--run id=eval_dir` で渡す。Teacher 行はいずれかの eval の image_metrics["teacher"] から取る。

効率（params/VRAM/レイテンシ）は特徴生成の重さ（WideResNet50 の有無等）を段単位で
正しく数える必要があるため、本スクリプトでは扱わず bin/bench_networks.py に委ねる。
ここは AUROC/AP/F1max と蒸留忠実度（map_mse/map_mae）の精度比較に集中する。
"""
import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import patchcore.runlog as runlog

plt.rcParams["font.family"] = "Noto Sans CJK JP"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 条件メタ（表示ラベルと構成）。id は ASCII、label は表示名。
META = {
    "T0":  {"label": "T0 Teacher", "input": "PatchCore", "train": "正常のみ",
            "feat": "WideResNet50", "back": "faiss"},
    "P1":  {"label": "P1", "input": "画像", "train": "正常のみ",
            "feat": "ResNet18(一体)", "back": "一体"},
    "P2":  {"label": "P2", "input": "画像", "train": "正常+疑似",
            "feat": "ResNet18(一体)", "back": "一体"},
    "P3":  {"label": "P3", "input": "特徴", "train": "正常のみ",
            "feat": "WideResNet50", "back": "学習decoder"},
    "P4":  {"label": "P4", "input": "特徴", "train": "正常+疑似",
            "feat": "WideResNet50", "back": "学習decoder"},
    "P5A": {"label": "③A", "input": "画像", "train": "正常+疑似",
            "feat": "ResNet18前段", "back": "②凍結"},
    "P5B": {"label": "③B", "input": "画像", "train": "正常+疑似",
            "feat": "ResNet18前段", "back": "再学習"},
}
ORDER = ["T0", "P1", "P2", "P3", "P4", "P5A", "P5B"]


def parse_args():
    p = argparse.ArgumentParser(description="蒸留最終比較（精度）の横断集計")
    p.add_argument("--run", action="append", default=[], metavar="id=eval_dir",
                   help="条件ID=eval出力フォルダ。例 P1=results/.../eval_FINAL_P1_...")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "distill_map"))
    p.add_argument("--name", default="final_compare3")
    return p.parse_args()


def _read_eval(eval_dir):
    im = json.load(open(os.path.join(eval_dir, "metrics", "image_metrics.json")))
    fid_path = os.path.join(eval_dir, "metrics", "distillation_fidelity.json")
    fid = json.load(open(fid_path)) if os.path.exists(fid_path) else {}
    return im, fid


def _row(cid, im, fid):
    m = META.get(cid, {"label": cid, "input": "-", "train": "-", "feat": "-", "back": "-"})
    s = im.get("student", {})
    return {
        "id": cid, "label": m["label"], "input": m["input"], "train": m["train"],
        "feat": m["feat"], "back": m["back"],
        "auroc": s.get("auroc"), "ap": s.get("average_precision"), "f1_max": s.get("f1_max"),
        "map_mse": fid.get("map_mse"), "map_mae": fid.get("map_mae"),
        "score_spearman": fid.get("score_spearman"),
    }


def _teacher_row(im):
    t = im.get("teacher", {})
    return {
        "id": "T0", "label": META["T0"]["label"], "input": META["T0"]["input"],
        "train": META["T0"]["train"], "feat": META["T0"]["feat"], "back": META["T0"]["back"],
        "auroc": t.get("auroc"), "ap": t.get("average_precision"), "f1_max": t.get("f1_max"),
        "map_mse": 0.0, "map_mae": 0.0, "score_spearman": 1.0,
    }


def _plot(rows, out_path):
    labels = [r["label"] for r in rows]
    x = range(len(rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    w = 0.26
    for k, off, name, color in (
        ("auroc", -w, "Instance AUROC", "#2ca02c"),
        ("ap", 0.0, "AP", "#1f77b4"),
        ("f1_max", w, "F1max", "#ff7f0e"),
    ):
        vals = [r[k] if r[k] is not None else 0.0 for r in rows]
        ax1.bar([i + off for i in x], vals, width=w, label=name, color=color)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=20)
    ax1.set_ylabel("スコア（↑ 高いほど良い）")
    ax1.set_title("異常スコア評価：Instance AUROC / AP / F1max")
    ax1.set_ylim(0, 1.0)
    ax1.axhline(rows[0]["auroc"], ls="--", color="gray", lw=1,
                label=f"Teacher AUROC={rows[0]['auroc']:.3f}")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    mse = [r["map_mse"] if r["map_mse"] else 1e-6 for r in rows]
    ax2.bar(list(x), mse, color="#d62728")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=20)
    ax2.set_ylabel("map_MSE（↓ 低いほど良い, 対数）")
    ax2.set_title("異常マップ忠実度：map_MSE")
    ax2.set_yscale("log")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = runlog.make_run_dir(args.out, timestamp=f"{args.name}_{runlog.jst_timestamp()}")
    runlog.start_tee(run_dir)
    print(f"出力先: {run_dir}")

    parsed = {}
    teacher_im = None
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"--run は id=eval_dir 形式: {spec}")
        cid, eval_dir = spec.split("=", 1)
        im, fid = _read_eval(eval_dir)
        parsed[cid] = _row(cid, im, fid)
        if teacher_im is None:
            teacher_im = im
        print(f"  {cid}: {eval_dir} → AUROC={parsed[cid]['auroc']}")

    rows = []
    if teacher_im is not None:
        rows.append(_teacher_row(teacher_im))
    for cid in ORDER:
        if cid in parsed and cid != "T0":
            rows.append(parsed[cid])
    for cid, r in parsed.items():  # ORDER 外の id も末尾に
        if cid not in ORDER:
            rows.append(r)

    cols = ["id", "label", "input", "train", "feat", "back",
            "auroc", "ap", "f1_max", "map_mse", "map_mae", "score_spearman"]
    csv_path = os.path.join(run_dir, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    _plot(rows, os.path.join(run_dir, "comparison_scores.png"))

    print("\n=== 最終比較（精度） ===")
    hdr = f"{'条件':<10}{'入力':<8}{'学習':<10}{'AUROC':>8}{'AP':>8}{'F1max':>8}{'map_MSE':>10}"
    print(hdr)
    for r in rows:
        au = f"{r['auroc']:.4f}" if r["auroc"] is not None else "-"
        ap = f"{r['ap']:.4f}" if r["ap"] is not None else "-"
        f1 = f"{r['f1_max']:.4f}" if r["f1_max"] is not None else "-"
        mm = f"{r['map_mse']:.5f}" if r["map_mse"] is not None else "-"
        print(f"{r['label']:<10}{r['input']:<8}{r['train']:<10}{au:>8}{ap:>8}{f1:>8}{mm:>10}")

    runlog.write_meta(run_dir, extra={"runs": args.run, "n_conditions": len(rows)})
    print(f"\n完了: {run_dir}")


if __name__ == "__main__":
    main()
