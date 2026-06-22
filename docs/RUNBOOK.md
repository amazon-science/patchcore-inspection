# RUNBOOK：スクリプト実行方法一覧

本リポジトリの全スクリプト（Python / bash）の実行方法を積み上げる索引。**常に最新の状態に保つこと。**

> **更新ルール**（詳細は `CLAUDE.md`「実行・結果出力の規約」を参照）
> - 新規スクリプトの追加・実行方法の変更は、**同じコミットで本ファイルを更新**する。
> - 各エントリは「目的 / 前提 / コマンド例 / 出力先 / 想定時間」を記載する。
> - 結果は **`YYYYMMDD_HHMMSS`（JST）** のタイムスタンプ付きで出力し、上書きしない。
> - 実験系の出力は **図・グラフを必ず伴い**、結果フォルダに `meta.json` と `log.txt` を残す。
> - 出力先は **`./results/` 配下**に限定（`./dataset/`・`/disk/data_area4/` は読み取り専用）。

共通の前提：
- venv を有効化（`source .venv/bin/activate`）、または `.venv/bin/python` を直接指定する。
- `bin/` 配下の直接実行時は **`PYTHONPATH=./src`** が必須。

---

## 学習・評価パイプライン

### bin/run_patchcore.py — PatchCore モデルの学習

- **目的**：Backbone からパッチ特徴を抽出し、Coreset 圧縮メモリバンクを構築・保存する。
- **前提**：`PYTHONPATH=./src`、GPU、`--faiss_on_gpu` は faiss-gpu ビルドが必要。
- **コマンド例**：
  ```bash
  bash bin/run_patchcore_venv.sh
  # または直接実行は CLAUDE.md「パイプライン全体の流れ」を参照
  ```
- **出力先**：`./results/<log_project>/<log_group>/`（`results.csv`・モデル）
- **想定時間**：データ規模・GPU 依存（長時間ジョブ。自動実行禁止）

### bin/load_and_evaluate_patchcore.py — 学習済みモデルの評価

- **目的**：保存済みモデルをロードして AUROC 等を算出する。
- **前提**：`PYTHONPATH=./src`、学習済みモデルのパス。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src .venv/bin/python bin/load_and_evaluate_patchcore.py \
    --gpu 0 --seed 0 <save_folder> \
    patch_core_loader -p <model_path> --faiss_on_gpu \
    dataset -d anomaly_dataset --resize 256 --imagesize 224 mvtec ./dataset
  ```
- **出力先**：`<save_folder>/results.csv`
- **想定時間**：評価データ規模依存（長時間ジョブ。自動実行禁止）

---

## 蒸留（distill）

### src/patchcore/distill/train_student.py — 生徒 MLP の学習（知識蒸留）

- **目的**：教師 faiss バンクへの距離回帰で生徒 MLP を学習し、メモリ削減を図る。
- **前提**：`PYTHONPATH=./src`、教師バンク特徴量。
- **コマンド例**：（呼び出し方法を確定後にここへ追記）
- **出力先**：（要記載）
- **想定時間**：（要記載）

---

## 比較・可視化スクリプト

### scripts/compare_distillation.py — 蒸留手法の比較プロット

- **目的**：`results.csv`（method, auroc, f1, memory_mb, fps）から AUROC×メモリ散布図・指標棒グラフを描く。
- **前提**：`PYTHONPATH=./src`、対象 run ディレクトリ。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src python scripts/compare_distillation.py <run_dir>
  ```
- **出力先**：`<run_dir>/comparison.png`
- **想定時間**：数秒
- **備考**：現状は出力ファイル名が固定（上書きされる）。規約2・7 への準拠は別タスクで改修予定。

### scripts/compare_continual_learning.py — 継続学習 P×R 比較プロット

- **目的**：`results/ContinualLearning/<timestamp>/results.csv` を読み、コアセット率 P × ラウンド数 R の 3×3 グリッド比較図を出力。
- **前提**：`PYTHONPATH=./src`、対象結果ディレクトリ。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src python scripts/compare_continual_learning.py
  ```
- **出力先**：（要記載：`results/ContinualLearning/<timestamp>/`）
- **想定時間**：数秒

### scripts/visualize_benchmark.py — ベンチマーク結果の可視化

- **目的**：ベンチマーク結果から比較図（PNG / HTML）を生成。
- **前提**：`results_dir` に集計済み結果。
- **コマンド例**：
  ```bash
  python scripts/visualize_benchmark.py [--results_dir RESULTS_DIR]
  ```
- **出力先**：`results/Benchmark_Results/benchmark_plot.png` / `benchmark_plot.html`
- **想定時間**：数秒

---

## ユーティリティ

### scripts/extract_rar.py — RAR アーカイブの展開

- **目的**：異常データセットの `.rar` を展開する。
- **前提**：`unrar` / `7z` / `bsdtar` のいずれかがインストール済み。
- **コマンド例**：
  ```bash
  # 単一ファイル
  python scripts/extract_rar.py /path/to/file.rar -o /path/to/output
  # ディレクトリ内を再帰展開
  python scripts/extract_rar.py /path/to/rar_dir -r -o /path/to/output
  ```
- **出力先**：`-o` で指定（既定はカレント）
- **想定時間**：アーカイブ規模依存

### scripts/experiment_done_hook.py — 実験完了フック（自動実行）

- **目的**：実験完了時に `results.csv` のサマリーを表示する PostToolUse フック。
- **前提**：Claude Code のフックとして自動起動（手動実行は通常不要）。

---

## テスト・静的チェック

```bash
# 構文チェック
python -m py_compile <file>

# テスト
PYTHONPATH=./src python3 -m pytest -v
# または
./local_run_test.sh
```
