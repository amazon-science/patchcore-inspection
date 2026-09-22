#!/usr/bin/env python3
"""疑似異常「形状／境界」実験の成果物を1か所に集約して保存する（再現性確保）。

図・疑似異常サンプル画像・指標・再現メタ（git commit / seed / 実行コマンド / 環境）を
`<out>/{figures,samples,metrics,repro}/` に整理してコピーし、README とマニフェストを書く。
生成は決定論的（seed＋コード commit＋引数で完全再現）なので、疑似画像本体は
サンプルのみを保存し、再現手順を repro/commands.md に残す方針。

書き込みは `--out` 配下のみ。入力（results/・pack）は読み取りのみ。
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys

import patchcore.runlog as runlog

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_RE = re.compile(r"^(?P<key>.+)_(?P<ts>\d{8}_\d{6})$")


README_HEAD = """# 疑似異常の形状・境界バリアント実験（アーカイブ）

PatchCore 蒸留（① image→map / B方式教師）で、**疑似異常画像の生成方法だけ**を差し替えた
対照実験の成果物一式。後段（教師モデル・学習条件・評価セット）は前回実験と同一。

- アーカイブ作成：{ts}（JST）
- git commit：`{commit}`
- 教師モデル：`results/ContinualLearning/20260526_比較検討/20260527_064537_N1000_P5/round_21`（Instance AUROC 0.9282）
- 評価セット：`results/distill_map/precomputed/20260622_222843_all`（test 1,388枚＝正常1,272＋異常116）

## ディレクトリ

| パス | 内容 |
|------|------|
| `figures/01_shape_distributions.png` | 疑似異常と実欠陥の形状分布（面積率/充填率/円形度/縦横比の CDF） |
| `figures/02_shape_samples.png` | 形状バリアントの見た目比較 |
| `figures/03_auroc_by_shape.png` | 条件別 AUROC（参照線：教師・inpainting926・前回10x） |
| `figures/04_score_alignment.png` | 疑似異常の教師スコア分布 vs 実異常（効いた理由） |
| `figures/05_shape_factorial.png` | 形 × 境界の 2×2 要因計画（スケール別・複数シード平均±SD） |
| `figures/conditions/<条件>/` | ROC・学習曲線・正常過学習検知・異常マップパネル |
| `samples/<バリアント>/` | 疑似異常のサンプル PNG・生成メタ・pack サニティ図 |
| `metrics/` | 横断表（summary.csv / comparison.csv / factorial.csv / shape_stats.csv / score_alignment.csv）と条件別指標 |
| `repro/env.json` | git commit・入力パス・条件一覧 |
| `repro/run_meta/<条件>/` | 学習・評価の meta.json（実行コマンド全文・seed・環境） |
| `repro/commands.md` | 生成→教師ターゲット→学習→評価の再現コマンド |
| `manifest.csv` | 保存ファイル一覧（サイズ・ハッシュ） |

## 結果（実行単位。条件ごとの平均±SD は `metrics/factorial.csv` を参照）

| 条件 | AUROC | AP | F1max |
|------|-------|----|----|
"""

README_TAIL = """
## 主要な結論

- **境界の描画方法（ベタ塗り hard → σ∝半径のぼかし soft）が精度を決める**：主効果 **+0.143（1x）／+0.140（10x）**。
- **形（凸多角形 poly → 実欠陥に整合させた不定形 irregular）の寄与は小さい**：**+0.016（1x）／−0.006（10x）**。
- 効いた理由：疑似異常の Teacher スコア中央値が 実異常 3.24 に対し hard 6.2／soft 4.1。
  ベタ塗りは教師から見て「異常すぎる」疑似を作っており、実異常へ転移しない（`figures/04_score_alignment.png`）。
- スケール（23,888→238,880枚）による改善はほぼ無い＝**枚数より疑似の質が支配的**。
- 詳細は `figures/05_shape_factorial.png` と `metrics/factorial.csv`。

## 再現方法

疑似異常の生成は **seed とコード commit と引数で決定論的**に再現できるため、画像本体は
サンプルのみを保存している。完全な再生成は `repro/commands.md` のコマンドを順に実行する。

1. 疑似異常画像の生成（`scripts/generate_pseudo_anomalies.py`）
2. B方式の正解マップ生成（`scripts/make_pseudo_targets_patchcore.py`、教師 PatchCore の NN 距離）
3. ① image_to_map の学習（`bin/train_distill_map.py`）
4. 評価（`bin/eval_distill_map.py`）

各条件の実行コマンド全文は `repro/run_meta/<条件>/{train,eval}_meta.json` の `command` にも残っている。
"""


def _write_readme(out, env, summary):
    rows = sorted(summary, key=lambda x: -x.get("auroc", 0))
    body = README_HEAD.format(ts=env["archived_at_jst"], commit=env["git_commit"])
    for r in rows:
        body += (f"| {r['condition']} | {r.get('auroc', float('nan')):.4f} | "
                 f"{r.get('ap', float('nan')):.4f} | {r.get('f1max', float('nan')):.4f} |\n")
    body += f"| （参考）Teacher PatchCore | {rows[0].get('teacher_auroc', float('nan')):.4f} | — | — |\n"
    body += README_TAIL
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(body)
    # 再現コマンド（各条件の meta.json から実コマンドを抜き出して並べる）
    lines = ["# 再現コマンド\n",
             "前提：`cd <repo>` / `export PYTHONPATH=./src` / venv 有効化。\n",
             "\n## 1. 疑似異常画像の生成（seed とこの引数で決定論的に再現される）\n"]
    sam = os.path.join(out, "samples")
    for variant in sorted(os.listdir(sam)) if os.path.isdir(sam) else []:
        for f, title in (("meta.json", "画像生成"), ("B_meta.json", "B方式 教師ターゲット生成")):
            mp = os.path.join(sam, variant, f)
            if os.path.exists(mp):
                with open(mp) as fh:
                    cmd = json.load(fh).get("command")
                if cmd:
                    lines.append(f"\n### {variant} — {title}\n```bash\n{cmd}\n```\n")
    lines.append("\n## 2. 学習と評価（条件ごと）\n")
    for r in rows:
        cond = r["condition"]
        lines.append(f"\n## {cond}\n")
        for tag in ("train", "eval"):
            mp = os.path.join(out, "repro", "run_meta", cond, f"{tag}_meta.json")
            if os.path.exists(mp):
                with open(mp) as f:
                    lines.append(f"```bash\n{json.load(f)['command']}\n```\n")
    with open(os.path.join(out, "repro", "commands.md"), "w") as f:
        f.writelines(lines)


def parse_args():
    p = argparse.ArgumentParser(description="形状実験の成果物アーカイブ")
    p.add_argument("--out", required=True, help="保存先ディレクトリ（ここ以外には書き込まない）")
    p.add_argument("--run_dir", nargs="+", required=True,
                   help="学習・評価の出力フォルダ（例 results/distill_map/shape_screen_1x）")
    p.add_argument("--shape_stats", default="", help="compare_pseudo_shape_stats の出力（既定=最新）")
    p.add_argument("--comparison", default="", help="aggregate_shape_comparison の出力（既定=最新）")
    p.add_argument("--score_alignment", default="", help="plot_pseudo_score_alignment の出力（既定=最新）")
    p.add_argument("--factorial", default="", help="plot_shape_factorial の出力（既定=最新）")
    p.add_argument("--extra_pack", nargs="+", default=[],
                   help="`バリアント名=生成 run フォルダ` の並び（別ストレージの pack のサンプルも回収）")
    p.add_argument("--pseudo_root", nargs="+", default=[],
                   help="疑似 pack の親（例 /tmp/t-hori/bfai/shape_screen_1x）。sample_png を回収する")
    p.add_argument("--n_samples", type=int, default=200, help="条件ごとに保存するサンプル PNG 枚数")
    p.add_argument("--n_panels", type=int, default=8, help="条件ごとに保存する異常マップパネル枚数")
    p.add_argument("--force", action="store_true", help="既存の out が空でなくても続行")
    return p.parse_args()


def _latest(base):
    ds = sorted(d for d in os.listdir(base) if TS_RE.match(d + "_00000000_000000") or re.match(r"^\d{8}_\d{6}$", d))
    return os.path.join(base, ds[-1]) if ds else None


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _sha256(path, limit=64 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if f.tell() > limit:  # 大きいファイルは先頭のみ（同一性の目安）
                break
    return h.hexdigest()


def discover(run_dirs):
    """eval フォルダを起点に条件を割り出し、対応する学習フォルダを結びつける。"""
    conds = {}
    for rd in run_dirs:
        if not os.path.isdir(rd):
            continue
        entries = sorted(os.listdir(rd))
        for name in entries:
            if not name.startswith("eval_"):
                continue
            m = TS_RE.match(name)
            if not m:
                continue
            cond = m.group("key")[len("eval_"):]
            conds.setdefault(cond, {})["eval"] = os.path.join(rd, name)
            conds[cond]["eval_ts"] = m.group("ts")
        for name in entries:
            m = TS_RE.match(name)
            if not m or name.startswith("eval_"):
                continue
            key = m.group("key")
            for cond in conds:
                rest = key[len(cond):] if key.startswith(cond) else None
                if rest in ("", "_image_to_map"):
                    conds[cond]["train"] = os.path.join(rd, name)
    return {k: v for k, v in sorted(conds.items()) if "eval" in v}


def main():
    args = parse_args()
    out = os.path.abspath(args.out)
    if os.path.isdir(out) and os.listdir(out) and not args.force:
        raise SystemExit(f"保存先が空ではありません（--force で続行可）: {out}")
    os.makedirs(out, exist_ok=True)
    runlog.start_tee(out)

    base = os.path.join(REPO_ROOT, "results", "distill_map")
    shape_stats = args.shape_stats or _latest(os.path.join(base, "shape_stats"))
    comparison = args.comparison or _latest(os.path.join(base, "shape_comparison"))
    alignment = args.score_alignment or _latest(os.path.join(base, "score_alignment"))
    factorial = args.factorial or _latest(os.path.join(base, "shape_factorial"))

    copied = []
    # --- 1) 主要図 ---
    top = [(os.path.join(shape_stats, "shape_distributions.png"), "figures/01_shape_distributions.png"),
           (os.path.join(shape_stats, "shape_samples.png"), "figures/02_shape_samples.png"),
           (os.path.join(comparison, "auroc_by_shape.png"), "figures/03_auroc_by_shape.png"),
           (os.path.join(alignment, "score_alignment.png"), "figures/04_score_alignment.png"),
           (os.path.join(factorial or "", "shape_factorial.png"), "figures/05_shape_factorial.png")]
    for src, rel in top:
        if os.path.exists(src):
            copied.append(_copy(src, os.path.join(out, rel)))

    # --- 2) 指標（横断表）---
    for src, rel in [(os.path.join(shape_stats, "shape_stats.csv"), "metrics/shape_stats.csv"),
                     (os.path.join(comparison, "comparison.csv"), "metrics/comparison.csv"),
                     (os.path.join(alignment, "score_alignment.csv"), "metrics/score_alignment.csv"),
                     (os.path.join(factorial or "", "factorial.csv"), "metrics/factorial.csv")]:
        if os.path.exists(src):
            copied.append(_copy(src, os.path.join(out, rel)))

    # --- 3) 条件ごとの図・指標・メタ ---
    conds = discover(args.run_dir)
    summary = []
    for cond, d in conds.items():
        ed, td = d["eval"], d.get("train")
        fig_dir = os.path.join(out, "figures", "conditions", cond)
        roc = os.path.join(ed, "roc_teacher_vs_student.png")
        if os.path.exists(roc):
            copied.append(_copy(roc, os.path.join(fig_dir, "roc.png")))
        vis = os.path.join(ed, "visualizations")
        if os.path.isdir(vis):
            for f in sorted(os.listdir(vis))[:args.n_panels]:
                copied.append(_copy(os.path.join(vis, f), os.path.join(fig_dir, "panels", f)))
        for f in ("train_curve.png", "overfit_normal.png"):
            if td and os.path.exists(os.path.join(td, f)):
                copied.append(_copy(os.path.join(td, f), os.path.join(fig_dir, f)))
        for f in ("image_metrics.json", "distillation_fidelity.json", "efficiency_metrics.json"):
            src = os.path.join(ed, "metrics", f)
            if os.path.exists(src):
                copied.append(_copy(src, os.path.join(out, "metrics", "conditions", cond, f)))
        if td:
            for f in ("train_log.csv", "config.json"):
                if os.path.exists(os.path.join(td, f)):
                    copied.append(_copy(os.path.join(td, f), os.path.join(out, "metrics", "conditions", cond, f)))
        for tag, src in (("eval", os.path.join(ed, "meta.json")),
                         ("train", os.path.join(td, "meta.json") if td else "")):
            if src and os.path.exists(src):
                copied.append(_copy(src, os.path.join(out, "repro", "run_meta", cond, f"{tag}_meta.json")))
        row = {"condition": cond, "eval_dir": ed, "train_dir": td}
        im = os.path.join(ed, "metrics", "image_metrics.json")
        if os.path.exists(im):
            with open(im) as f:
                j = json.load(f)
            row.update({"auroc": j["student"]["auroc"], "ap": j["student"]["average_precision"],
                        "f1max": j["student"]["f1_max"], "teacher_auroc": j["teacher"]["auroc"]})
        summary.append(row)

    # --- 4) 疑似異常のサンプル画像と生成メタ ---
    for root in args.pseudo_root:
        if not os.path.isdir(root):
            continue
        for variant in sorted(os.listdir(root)):
            vdir = os.path.join(root, variant)
            if not os.path.isdir(vdir):
                continue
            for ts in sorted(os.listdir(vdir)):
                sp = os.path.join(vdir, ts, "sample_png")
                if not os.path.isdir(sp):
                    continue
                for f in sorted(os.listdir(sp))[:args.n_samples]:
                    copied.append(_copy(os.path.join(sp, f), os.path.join(out, "samples", variant, f)))
                for f in ("pack_sanity.png", "meta.json", "log.txt"):
                    src = os.path.join(vdir, ts, f)
                    if os.path.exists(src):
                        copied.append(_copy(src, os.path.join(out, "samples", variant, f)))
            for f, dst in (("packed_meta.json", "B_packed_meta.json"), ("meta.json", "B_meta.json")):
                src = os.path.join(root, f"{variant}_B", f)
                if os.path.exists(src):
                    copied.append(_copy(src, os.path.join(out, "samples", variant, dst)))

    # --- 4b) 別ストレージにある pack のサンプル（前回方式の疑似など）---
    for kv in args.extra_pack:
        variant, _, rdir = kv.partition("=")
        sp = os.path.join(rdir, "sample_png")
        if os.path.isdir(sp):
            for f in sorted(os.listdir(sp))[:args.n_samples]:
                copied.append(_copy(os.path.join(sp, f), os.path.join(out, "samples", variant, f)))
        for f in ("pack_sanity.png", "meta.json"):
            if os.path.exists(os.path.join(rdir, f)):
                copied.append(_copy(os.path.join(rdir, f), os.path.join(out, "samples", variant, f)))

    # --- 5) 再現メタ ---
    env = {"git_commit": runlog.git_commit_hash(), "archived_at_jst": runlog.jst_timestamp(),
           "repo_root": REPO_ROOT, "run_dirs": args.run_dir, "pseudo_roots": args.pseudo_root,
           "shape_stats": shape_stats, "comparison": comparison, "score_alignment": alignment,
           "factorial": factorial,
           "conditions": summary}
    with open(os.path.join(out, "repro", "env.json"), "w") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out, "metrics", "summary.csv"), "w") as f:
        f.write("condition,auroc,ap,f1max,teacher_auroc,eval_dir\n")
        for r in sorted(summary, key=lambda x: -x.get("auroc", 0)):
            f.write(f"{r['condition']},{r.get('auroc','')},{r.get('ap','')},{r.get('f1max','')},"
                    f"{r.get('teacher_auroc','')},{r['eval_dir']}\n")

    man = os.path.join(out, "manifest.csv")
    with open(man, "w") as f:
        f.write("path,bytes,sha256_head64mb\n")
        for p in sorted(copied):
            rel = os.path.relpath(p, out)
            f.write(f"{rel},{os.path.getsize(p)},{_sha256(p)}\n")

    _write_readme(out, env, summary)

    print(f"条件 {len(conds)} 件 / ファイル {len(copied)} 本を保存: {out}")
    for r in sorted(summary, key=lambda x: -x.get("auroc", 0)):
        print(f"  {r['condition']:24s} AUROC={r.get('auroc', float('nan')):.4f}")


if __name__ == "__main__":
    sys.exit(main())
