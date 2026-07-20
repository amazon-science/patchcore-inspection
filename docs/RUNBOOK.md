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
- **出力先**：`./results/<log_project>/<log_group>/`（`iterate` で `_0/_1` を付与し上書き防止）
  - `results.csv` / `results_summary.png`（指標サマリ図・自動生成）/ `meta.json` / `log.txt`
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
- **出力先**：`<save_folder>/`（`results.csv` / `results_summary.png` / `meta.json` / `log.txt`）
- **想定時間**：評価データ規模依存（長時間ジョブ。自動実行禁止）

### bin/run_continual_learning.py — 継続学習実験

- **目的**：各ラウンドで一部画像を学習しコアセットをマージしていく継続学習を評価する。
- **前提**：`PYTHONPATH=./src`、GPU、`./dataset`。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src .venv/bin/python bin/run_continual_learning.py \
    --n_per_round 2000 --percentage 0.10 --n_rounds 10
  ```
- **出力先**：`results/ContinualLearning/<YYYYMMDD_HHMMSS>/`
  - `results.csv` / `progress.png` / `round_NN/`（チェックポイント）/ `meta.json` / `log.txt`
- **想定時間**：ラウンド数・枚数依存（長時間ジョブ。自動実行禁止）

### bin/benchmark_scaling.sh — スケーリングベンチマーク

- **目的**：画像枚数 × サブサンプリング率を振って train_time / memory_mb を計測する（run_patchcore のラッパ）。
- **前提**：Python 3.10 実バイナリ、`./dataset`。
- **コマンド例**：
  ```bash
  bash bin/benchmark_scaling.sh [GPU番号]
  ```
- **出力先**：`results/Benchmark_Results/<YYYYMMDD_HHMMSS>/`
  - `Bench_*/`（各条件、`results.csv`＋`meta.json`＋`log.txt`）/ `summary.csv` / `meta.json` / `log.txt`
  - `viz/<timestamp>/`（完了時に visualize_benchmark.py が自動生成）
- **想定時間**：条件数依存（全条件で 30〜60 分程度。長時間ジョブ。自動実行禁止）

---

## 蒸留（distill）

### bin/run_distillation.py — 蒸留手法の比較ハーネス（エントリポイント）

- **目的**：教師 PatchCore のメモリバンクを M0〜M4 の軽量表現へ置き換え、AUROC×メモリ×FPS で比較する。
- **前提**：`PYTHONPATH=./src`、教師モデルフォルダ（`patchcore_params.pkl` を含む）、GPU。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src .venv/bin/python bin/run_distillation.py <teacher_dir> \
    --gpu 0 --seed 0 --methods M0 M1 M2 M3
  ```
- **出力先**：`results/Distillation/<YYYYMMDD_HHMMSS>/`（`--out` で親を変更可）
  - `results.csv` / `comparison.png`（自動生成）/ `meta.json` / `log.txt`
- **想定時間**：手法数・テスト枚数・生徒学習ステップ数依存（M3/M4 を含むと数分〜）

### src/patchcore/distill/train_student.py — 生徒 MLP の学習（ライブラリ）

- **位置づけ**：単体実行しない**ライブラリモジュール**。`train_student()` を `bin/run_distillation.py`（M3/M3L/M3R）が呼び出す。
- **出力先**：呼び出し元（`run_distillation.py`）の `results/Distillation/<timestamp>/` に集約される。

---

## マップ蒸留（distill_map）

> 既存の distill（M0-M4 スコア蒸留）とは**別軸の独立実験**。ResNet18-FPN/Decoder
> Student で PatchCore の異常マップ A_T・スコア s_T を dense 予測として蒸留する。
> 設計：`meeting_logs/PatchCore_知識蒸留_実装設計書.md` / 計画：`meeting_logs/roadmap/roadmap_20260622.md`。

### src/patchcore/distill_map/ — マップ蒸留の共通モジュール（ライブラリ）

- **位置づけ**：単体実行しない**ライブラリ**。`pooling`（map→score）、`losses`（map/score 蒸留 loss）、`student_fpn`（ResNet18+FPN+Decoder, Pattern B）等を提供し、`bin/` のエントリから呼ばれる。
- **前提**：`PYTHONPATH=./src`。
- **動作確認**：`PYTHONPATH=./src .venv/bin/python -c "from patchcore.distill_map import pooling, losses, student_fpn"`

### scripts/precompute_teacher_maps.py — Teacher 異常マップ/スコアの事前計算

- **目的**：学習済み PatchCore を 1 度 forward し、各サンプルの A_T（異常マップ）・s_T（スコア）を `.pt` 保存（蒸留の教師信号）。正規化は学習時に行うため**生のまま保存**。
- **前提**：`PYTHONPATH=./src`、教師フォルダ（`patchcore_params.pkl`）、GPU。保護対象データは読み取りのみ。
- **コマンド例**：
  ```bash
  # スモーク（各 split 4 枚）
  PYTHONPATH=./src .venv/bin/python scripts/precompute_teacher_maps.py <teacher_dir> \
    --gpu 0 --seed 0 --splits test train --max_per_split 4
  # 本実行
  PYTHONPATH=./src .venv/bin/python scripts/precompute_teacher_maps.py <teacher_dir> \
    --gpu 0 --seed 0 --splits train test
  # Pattern A 用（特徴マップ F_T も保存。.pt が大きくなる）
  PYTHONPATH=./src .venv/bin/python scripts/precompute_teacher_maps.py <teacher_dir> \
    --gpu 0 --seed 0 --splits train test --save_feature
  ```
  - `--save_feature` で各 `.pt` に `teacher_feature_map`（[C,H',W'] float16, 例 [1024,28,28]）を追加。
  ```bash
  # 疑似異常画像フォルダに対して Teacher 出力を precompute（S3 用）
  PYTHONPATH=./src .venv/bin/python scripts/precompute_teacher_maps.py <teacher_dir> \
    --gpu 0 --image_dir results/distill_map/pseudo/<ts>/anomaly --image_dir_split pseudo
  ```
  - `--image_dir` は任意画像フォルダを処理し `<run>/pseudo/*.pt` に保存（image_path はフルパス）。
  ```bash
  # 疑似スケール実験①用：Teacher をスキップし描画マスクを直接ターゲット化（faiss NN 不要・高速）
  # packed memmap 出力（大規模向け。学習は PackedTargetDataset が自動使用）
  PYTHONPATH=./src .venv/bin/python scripts/precompute_teacher_maps.py \
    --map_from_mask results/distill_map/pseudo/<ts>/masks \
    --image_dir     results/distill_map/pseudo/<ts>/anomaly \
    --image_dir_split pseudo --mask_blur_sigma 2.0 --pack
  ```
  - `--map_from_mask`：`teacher` 不要。ぼかしマスク（σ=`--mask_blur_sigma`）を teacher_anomaly_map に、score=マップ最大値、label=1。`--pack` で `<run>/pseudo/` に packed memmap（images uint8 / targets fp16 / scores / labels / `packed_meta.json`）。省略時は per-file `.pt`。`mask_targets_sanity.png` を出力。
  - ②③（特徴が必要な pattern）は `--map_from_mask --save_feature` が未対応（今後追加）。①(image_to_map) 専用。
- **出力先**：`results/distill_map/precomputed/<YYYYMMDD_HHMMSS>/`
  - `<split>/<sample_id>.pt`（image_path / teacher_anomaly_map / teacher_anomaly_score / label / mask）
  - `stats.json`（スコア統計・map サイズ）/ `teacher_targets_sanity.png` / `meta.json` / `log.txt`
- **想定時間**：枚数・バンクサイズ依存（train 全枚数だと長時間。スモークは数十秒）

### bin/precompute_student_features.py — 凍結前段の出力 F_S を事前計算（③-B 高速化）

- **目的**：③-B で後段を F_S で再学習する際、凍結前段(image_to_feature)を毎エポック回すのを避けるため、**F_S を一度だけ計算して保存**する。前段は凍結＋学習時データ拡張なし（決定的変換）なので F_S はエポック不変＝キャッシュは完全等価。生成物は既存 precompute と同形式（`teacher_feature_map` を F_S に差し替え、A_T 等は元 Teacher 値を継承）で、後段は既存 `feature_to_map` 経路（backbone forward 不要）で高速学習できる。
- **前提**：`PYTHONPATH=./src`、前段 `best.ckpt`、元 precompute（train/ ＋ 任意で疑似）、GPU。
- **コマンド例**：
  ```bash
  # スモーク（各 split 20 枚）
  PYTHONPATH=./src .venv/bin/python bin/precompute_student_features.py \
    --front_checkpoint <P5_run>/checkpoints/best.ckpt \
    --targets_dir <allfeat_precompute> --pseudo_dir <pseudo_precompute>/pseudo \
    --gpu 0 --max_per_split 20
  # 本実行
  PYTHONPATH=./src .venv/bin/python bin/precompute_student_features.py \
    --front_checkpoint <P5_run>/checkpoints/best.ckpt \
    --targets_dir <allfeat_precompute> --pseudo_dir <pseudo_precompute>/pseudo --gpu 0
  ```
- **出力先**：`results/distill_map/precomputed_student_feat/<name>_<YYYYMMDD_HHMMSS>/{train,pseudo}/*.pt` ＋ `meta.json` / `log.txt`
- **後続**：`bin/train_distill_map.py <出力> --pseudo_dir <出力>/pseudo --pattern feature_to_map`（＝③-B 後段学習）。評価は `eval_distill_map.py --front_checkpoint <P5> --checkpoint <③-B後段>`。
- **想定時間**：前段1パス（数分〜十数分、枚数依存）

### bin/train_distill_map.py — Pattern B Student の蒸留学習（エントリポイント）

- **目的**：precompute 済み Teacher target を教師信号に、画像→異常マップ Student を学習（`L = λ_map·L_map + λ_score·L_score`）。
- **前提**：`PYTHONPATH=./src`、precompute 出力フォルダ（`train/` を含む）、GPU。画像前処理は precompute と一致（同一 MVTecDataset）。
- **コマンド例**：
  ```bash
  # スモーク（少枚数・少 epoch）
  PYTHONPATH=./src .venv/bin/python bin/train_distill_map.py <precompute_dir> \
    --name SMOKE --gpu 0 --pretrained 0 --epochs 3 --batch_size 2 --val_ratio 0.25
  # 本実行（S2: Pattern B = image→map, normal only）
  PYTHONPATH=./src .venv/bin/python bin/train_distill_map.py <precompute_dir> \
    --name S2_image_to_map --gpu 0 --epochs 100 --batch_size 32 \
    --lambda_map 1.0 --lambda_score 0.1 --normalize minmax
  # S1: Pattern A = feature→map（precompute は --save_feature 必須）
  PYTHONPATH=./src .venv/bin/python bin/train_distill_map.py <precompute_dir> \
    --name S1_feature_to_map --pattern feature_to_map --gpu 0 --epochs 100 --batch_size 32
  ```
  - `--pattern feature_to_map` は backbone なしで F_T→A_S を学習（軽量）。eval は checkpoint の pattern を自動判別。
  ```bash
  # ③前段: Pattern image_to_feature = 画像→Teacher特徴 F_T を MSE 回帰（precompute は --save_feature 必須）
  PYTHONPATH=./src .venv/bin/python bin/train_distill_map.py <normal_precompute_dir> \
    --pseudo_dir <pseudo_precompute_dir>/pseudo \
    --name P5_image_to_feature --pattern image_to_feature --gpu 0 --epochs 100 --batch_size 32
  ```
  - `--pattern image_to_feature`（③前段）：画像→`[C,H',W']` 特徴を **生F_T へ直接 MSE 回帰**（map/score 損失は不使用、`--normalize` 無関係）。出力活性化は `--feat_output_activation`（既定 `relu`＝post-ReLU 非負 F_T に整合）。学習曲線は feat MSE を描画。出力仕様は ②後段 `feature_to_map`（in_channels=1024・格子28×28）と一致し、凍結再利用（方針A）できる。
  ```bash
  # S3: normal + pseudo anomaly 結合学習（--pseudo_dir に pseudo precompute の split を渡す）
  PYTHONPATH=./src .venv/bin/python bin/train_distill_map.py <normal_precompute_dir> \
    --pseudo_dir <pseudo_precompute_dir>/pseudo --name S3_image_to_map --gpu 0 --epochs 30
  ```
  - `--pseudo_dir` 指定時は normal(TeacherTargetDataset) と pseudo を結合。pseudo フォルダが packed（`packed_meta.json` あり）なら `PackedTargetDataset`、そうでなければ `DirectImageTargetDataset` を自動選択。正規化統計は両者から算出。
  - **正常過学習の検知**：normal/pseudo をそれぞれ train と **hold-out（未使用・非オーバーサンプル）** に分割し、train 正常(probe) vs hold-out 正常の map 損失ギャップと hold-out 正常スコア分布（collapse 検知）を毎 epoch 記録。`overfit_normal.png` を出力、`train_log.csv` に `train_normal_map/holdout_normal_map/overfit_gap/holdout_normal_score_mean` 等。early stopping は hold-out（正常＋疑似）損失を監視。
  - **比率固定サンプラ**：`--normal_pseudo_ratio`（既定 1.0=1:1）でバッチ内 normal:pseudo を一定化。`--steps_per_epoch>0` で 1epoch のステップ数を固定（正常過反復による肥大化を抑制）。
  - **スケール切り出し**：`--pseudo_max N` で pseudo の先頭 N 枚のみ使用（100倍プールから 5/10/50 倍を再生成なしで切り出す）。
  - **高速化**：`--cache_normal 1`（正常 RAM 常駐, 既定ON）、pin_memory/persistent_workers/`--prefetch_factor`/`--channels_last`/cudnn.benchmark。`--probe_size` は過学習ギャップ用に評価する train 正常の枚数。
- **出力先**：`results/distill_map/<name>_<YYYYMMDD_HHMMSS>/`
  - `checkpoints/{best,last}.ckpt` / `train_curve.png` / `train_log.csv` / `overfit_normal.png` / `config.json` / `meta.json` / `log.txt`
- **想定時間**：枚数・epoch 依存（train 全枚数×100ep は長時間）

### bin/run_pseudo_scale_experiment.sh — 疑似異常スケール実験①（5/10/50/100倍）一括

- **目的**：正常23,888基準で疑似異常を 5/10/50/100 倍に増やし ①(image_to_map) の精度変化を比較。100倍プールを1回生成・pack し、5/10/50 倍は `--pseudo_max` で先頭サブセット参照（再生成なし）。
- **前提**：`PYTHONPATH`（スクリプト内で設定）、venv、GPU、教師フォルダ。**長時間ジョブ**（100倍生成＋pack で数百GB／数時間、学習×4）。実行はユーザー判断で。
- **コマンド例**：
  ```bash
  bash bin/run_pseudo_scale_experiment.sh <teacher_dir> [results/distill_map/pseudo_scale]
  ```
- **フロー**：正常/評価 Teacher precompute → blob 100倍生成 → map_from_mask --pack → 各スケール学習。
- **出力先**：`results/distill_map/pseudo_scale/<YYYYMMDD_HHMMSS>/{normal_pc,pseudo_gen,pseudo_pack,train}/`
- **後続**：`eval_distill_map`（各 best.ckpt）→ `aggregate_distill_final` → `render_results_table`（スケール別表）。

### scripts/generate_pseudo_anomalies.py — 疑似異常画像の生成（S3 用）

- **目的**：正常画像に合成欠陥（scratch/spot/particle/contamination/brightness/blur/cutpaste/**blob**）を載せた疑似異常画像とマスクを生成。後段で Teacher precompute → Student 蒸留学習（normal+pseudo）に使う。
- **前提**：`PYTHONPATH=./src`。GPU 不要。`src/patchcore/distill_map/pseudo_anomaly.py` を利用（`blob` は PIL.ImageDraw で cv2 非依存。既存 7 種は cv2 があれば高品質、無ければ numpy フォールバック）。
- **コマンド例**：
  ```bash
  # 既存 7 種を混合生成
  PYTHONPATH=./src .venv/bin/python scripts/generate_pseudo_anomalies.py \
    --num 1000 --seed 0

  # blob（雑な図形の大量描画）単独生成。サイズ・個数の上限を制御（スケール実験 S3 用）
  PYTHONPATH=./src .venv/bin/python scripts/generate_pseudo_anomalies.py \
    --num 1000 --seed 0 --types blob \
    --blob_count_min 3 --blob_count_max 15 \
    --blob_size_min 0.02 --blob_size_max 0.30 \
    --blob_shapes tri rect penta hexa circle ellipse
  ```
  - `blob`：三角/矩形/五角/六角/円/楕円を 1 画像に `[count_min,count_max]` 個、短辺比 `[size_min,size_max]` のサイズでランダム描画。マスクは描画領域。`blob_cfg` は meta.json に記録。
  - **既定値は実 test 欠陥マスク(116枚)の実測分布に整合**：欠陥数/画像 中央値1・p90=3 → `count 1〜3`／半径r p5=0.005・中央値0.022・p90=0.123 → `size 0.005〜0.13` を `--blob_size_dist log`（対数一様）でサンプル。test の欠陥は単色なので `--blob_color_mode gray`（白〜黒の単色, `--blob_gray_min/max`）が既定。`rgb` でランダム色も可。
  ```bash
  # 大規模（100倍）：シャード化サブフォルダ＋並列。--out に外部ストレージを指定可
  PYTHONPATH=./src .venv/bin/python scripts/generate_pseudo_anomalies.py \
    --num 2388800 --seed 0 --types blob \
    --blob_count_min 1 --blob_count_max 3 --blob_size_min 0.02 --blob_size_max 0.50 \
    --shard_size 5000 --workers 16 \
    --out /disk/data_area4/t-hori/bfai/data_augmentation
  ```
  - `--shard_size`：1シャード(サブフォルダ `anomaly/00000/` 等)あたり枚数。単一ディレクトリ過密を回避。`--workers`：並列プロセス数。ファイル名は全体 index `pseudo_{i:08d}.png`。
  - `--per_image N`：**ソース基点モード**（正常1枚につき N 枚を連番生成、各ソースを1回だけ読む＝I/O最小）。総数=正常枚数×N。ランダム抽出（`--num`）より高速。
  ```bash
  # 【推奨・大規模】直接 packed 出力：PNG を介さず image+ぼかしターゲットを memmap へ
  # （NFS で数百万 PNG のメタデータ律速を回避。①生成と②map_from_mask を統合）
  PYTHONPATH=./src .venv/bin/python scripts/generate_pseudo_anomalies.py \
    --source_dir dataset/anomaly_dataset/train/good \
    --per_image 100 --pack --mask_blur_sigma 2.0 --sample_png 500 \
    --seed 0 --types blob --blob_count_min 1 --blob_count_max 3 \
    --blob_size_min 0.02 --blob_size_max 0.50 --workers 14 \
    --out /disk/data_area4/t-hori/bfai/data_augmentation
  ```
  - `--pack`（`--per_image` 必須）：`<run>/pseudo/` に PackedWriter で `images(uint8 224)/targets(fp16 224 ぼかしマスク)/scores/labels/packed_meta.json` を直接生成。学習は `--pseudo_dir <run>/pseudo`（PackedTargetDataset が自動使用、`--pseudo_max` でスケール切出し）。閲覧用に `sample_png/` と `pack_sanity.png` も保存。map_from_mask は不要。
  - fork 共有 memmap に複数プロセスが分散書込み（NFS 永続化確認済み）。600GB 級は NFS 帯域律速で数時間。
- **出力先**：`<out>/<YYYYMMDD_HHMMSS>/`
  - PNG 経路：`anomaly/<shard>/pseudo_*.png` / `masks/<shard>/pseudo_*.png` / `pseudo_sanity.png`
  - pack 経路：`pseudo/{images,targets,scores,labels}.npy` / `packed_meta.json` / `sample_png/` / `pack_sanity.png`
  - 共通：`meta.json` / `log.txt`
- **想定時間**：1,000 枚で数十秒。100倍(239万枚)は pack で NFS 帯域(≈70MB/s)律速の数時間

### bin/eval_distill_map.py — Student の評価（エントリポイント）

- **目的**：学習済み Student を test set で推論し、image-level metrics・蒸留忠実度・効率指標・可視化・scores.csv/json を出力。`--teacher` 指定で Teacher 効率（memory bank search 込み）も計測し T0 vs S2 比較。
- **前提**：`PYTHONPATH=./src`、precompute 出力（`test/` を含む）、Student の `best.ckpt`、GPU。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src .venv/bin/python bin/eval_distill_map.py <precompute_dir> \
    --checkpoint <run>/checkpoints/best.ckpt --name eval_S2 --gpu 0 \
    --num_vis 8 --teacher <teacher_dir>
  # ③A 連結評価: 前段(image_to_feature) → 後段(feature_to_map, 凍結) を連結して評価
  PYTHONPATH=./src .venv/bin/python bin/eval_distill_map.py <precompute_dir> \
    --checkpoint <P4_feature_to_map_run>/checkpoints/best.ckpt \
    --front_checkpoint <P5_image_to_feature_run>/checkpoints/best.ckpt \
    --name eval_P5A --gpu 0 --num_vis 8
  ```
  - `--front_checkpoint` 指定時：`--checkpoint` は後段(feature_to_map)、`--front_checkpoint` は前段(image_to_feature)を指す。画像→前段→F_S→後段→異常マップ を連結し、正規化は後段由来。効率は合成モデル全体（前段+後段）を集計。方針B（後段をF_Sで再学習）でも後段ckptを差し替えるだけで同IF。
- **出力先**：`results/distill_map/<name>_<YYYYMMDD_HHMMSS>/`
  - `metrics/{image_metrics,distillation_fidelity,efficiency_metrics}.json`
  - `predictions/scores.csv` / `roc_teacher_vs_student.png` / `visualizations/*_panel.png` / `meta.json` / `log.txt`
- **想定時間**：test 枚数依存（数十秒〜数分）

### bin/run_final_distill_comparison.sh — 最終比較実験（2x2）の一括実行ハーネス

- **目的**：マップ蒸留の最終比較 4 条件（**差分は `--pattern` と `--pseudo_dir` の 2 軸のみ**、他ハイパラ完全統一）を train → eval まで一括実行し、横断集計（`summary.csv` / `summary_auroc.png`）まで出力する。
  - P1: `image_to_map` / 正常のみ、P2: `image_to_map` / 正常+疑似、P3: `feature_to_map` / 正常のみ、P4: `feature_to_map` / 正常+疑似
- **前提**：`.venv`、precompute 済みの正常ターゲット（map+feature 両保持）・疑似ターゲット・Teacher フォルダ。GPU 2 枚（`GPU_A`=image系 / `GPU_B`=feature系を並行、各レーン内は直列）。**長時間ジョブのため手動実行のみ**。
- **データセット**：正常/test=`dataset/anomaly_dataset`、疑似異常=公式 `dataset/inpainting_dataset/train/abnormal` 由来の precompute、Teacher=`results/ContinualLearning/.../round_21`。
- **コマンド例**：
  ```bash
  # 既定値（ALLFEAT/PSEUDO/TEACHER/GPU 等）で実行
  bash bin/run_final_distill_comparison.sh
  # 環境変数で上書き（例：GPU・epoch・precompute 差し替え）
  GPU_A=0 GPU_B=1 EPOCHS=30 \
    ALLFEAT=results/distill_map/precomputed_allfeat/<ts> \
    PSEUDO=results/distill_map/precomputed_pseudo_official/<ts>/pseudo \
    TEACHER=results/ContinualLearning/.../round_21 \
    bash bin/run_final_distill_comparison.sh
  ```
- **統一ハイパラ**：seed=42 / epochs=30(early_stopping patience=5) / batch=32 / lr=1e-4 / minmax / smooth_l1 / λmap=1.0 λscore=0.1 / weighted_map α=5.0 / pooling=max / fpn=decoder=128 / freeze_backbone=0（環境変数で一部上書き可）。
- **出力先**：`results/distill_map/FINAL_comparison_<YYYYMMDD_HHMMSS>/`
  - `summary.csv`（Teacher 込み 5 行：AUROC/AP/F1max/FPS/params/GPUメモリ）/ `summary_auroc.png` / `manifest.txt` / `orchestrator.log` / `<name>.log`
  - 各 train/eval は従来どおり `results/distill_map/{FINAL_Px_*,eval_FINAL_Px_*}_<ts>/` に隔離出力。
- **想定時間**：4 条件分の train（各 epoch×全枚数）＋ eval。GPU 2 レーン並行で数時間規模。

---

### bin/bench_networks.py — ネットワーク単位の効率ベンチマーク

- **目的**：比較条件を「特徴生成ネット」「異常生成ネット」の 2 段に分解し、各ネットワークの **パラメータ数・モデルサイズ(MB) / 推論ピークVRAM(MB) / レイテンシ(ms/枚, batch=1)** を計測する。pipeline 全体ではなく段ごとの内訳を得るのが目的。
  - T0(Teacher)=WideResNet50 embed + faiss NN 探索 / P1,P2(image_to_map)=ResNet18 backbone + FPN+head / P3,P4(feature_to_map)=WideResNet50 embed(Teacher共有) + FeatureToMap decoder / ③=ResNet18-FPN 前段 + FeatureToMap decoder（`--p5_front`＋`--p5b_back`）
  - faiss は torch 外メモリのため peak VRAM は計測不可 → bank index サイズ(vec×dim×4B)を `model_size_mb` に計上し `note` に明記。
- **前提**：`PYTHONPATH=./src`、`.venv`、GPU 1 枚。Teacher フォルダと P1–P4 の `best.ckpt`（③は前段/後段 ckpt）。短時間ジョブ（学習なし）。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src .venv/bin/python bin/bench_networks.py --gpu 0 \
    --teacher results/ContinualLearning/.../round_21 \
    --p1 results/distill_map/FINAL_P1_image_normal_<ts>/checkpoints/best.ckpt \
    --p2 results/distill_map/FINAL_P2_image_pseudo_<ts>/checkpoints/best.ckpt \
    --p3 results/distill_map/FINAL_P3_feature_normal_<ts>/checkpoints/best.ckpt \
    --p4 results/distill_map/FINAL_P4_feature_pseudo_<ts>/checkpoints/best.ckpt \
    --p5_front results/distill_map/P5_image_to_feature_<ts>/checkpoints/best.ckpt \
    --p5b_back results/distill_map/P5B_back_on_FS_<ts>/checkpoints/best.ckpt
  ```
- **出力先**：`results/distill_map/bench_networks_<YYYYMMDD_HHMMSS>/`
  - `networks.csv`（段×ネットワークの全指標）/ `bench_model_size.png` / `bench_peak_vram.png` / `bench_latency.png` / `meta.json` / `log.txt`
- **想定時間**：数十秒〜1 分（`--n` 反復回数に依存、既定 50）。

---

## 比較・可視化スクリプト

### bin/aggregate_distill_final.py — 最終比較（精度）の横断集計（T0/P1–P4/③A/③B）

- **目的**：各条件の eval 出力（`metrics/image_metrics.json`・`distillation_fidelity.json`）から AUROC/AP/F1max と map_MSE/MAE を1表・1図に集約。③A/③B を P1–P4 と横並びにする。効率は段の重さを正しく数える必要があるため扱わず [bench_networks.py](bin/bench_networks.py) に委ねる。
- **前提**：`PYTHONPATH=./src`、各条件の eval ディレクトリ（`eval_distill_map.py` が生成）。学習・GPU 不要。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src .venv/bin/python bin/aggregate_distill_final.py \
    --run P1=<eval_FINAL_P1_...> --run P2=<eval_FINAL_P2_...> \
    --run P3=<eval_FINAL_P3_...> --run P4=<eval_FINAL_P4_...> \
    --run P5A=<eval_P5A_frozen_...> --run P5B=<eval_P5B_retrained_...>
  ```
  - `--run id=eval_dir`。id は T0/P1/P2/P3/P4/P5A(=③A)/P5B(=③B)。Teacher 行はいずれかの eval の `image_metrics["teacher"]` から自動抽出。
- **出力先**：`results/distill_map/final_compare3_<YYYYMMDD_HHMMSS>/`
  - `comparison.csv`（id/label/入力/学習/特徴生成/後段/auroc/ap/f1_max/map_mse/map_mae/score_spearman）/ `comparison_scores.png` / `meta.json` / `log.txt`
- **想定時間**：数秒

### bin/compare_condition_panels.py — 条件横断の異常マップ比較グリッド（③A/③B込み）

- **目的**：行＝test 異常サンプル、列＝入力/Teacher A_T/P1–P4/③A/③B の異常マップ A_S を1枚のグリッドに並べる。各マップは個別に [0,1] 正規化して jet 描画し、局在の再現性を条件間で見比べる。A_S は PNG からの抽出でなく各モデルを再推論して生成。
- **前提**：`PYTHONPATH=./src`、precompute（test/）、P1–P4＋前段(P5)＋③B後段(P5B) の `best.ckpt`、GPU。③A の後段は `--p4` を凍結流用。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src .venv/bin/python bin/compare_condition_panels.py <precompute_dir> \
    --p1 <P1>/checkpoints/best.ckpt --p2 <P2>/checkpoints/best.ckpt \
    --p3 <P3>/checkpoints/best.ckpt --p4 <P4>/checkpoints/best.ckpt \
    --front <P5_front>/checkpoints/best.ckpt --p5b_back <P5B_back>/checkpoints/best.ckpt \
    --gpu 0 --num 8
  ```
- **出力先**：`results/distill_map/compare_conditions3_<YYYYMMDD_HHMMSS>/compare_6conditions.png` ＋ `meta.json` / `log.txt`
- **想定時間**：数十秒（num×条件数の推論）

### bin/render_results_table.py — 結果表をスライド体裁の PNG で描画

- **目的**：`aggregate_distill_final.py` の comparison.csv（精度）と `bench_networks.py` の networks.csv（性能=pipeline合計）を結合し、緑ヘッダ・グループ見出し（異常マップ評価/異常スコア評価/パフォーマンス）のスライド用結果表 PNG を生成。P5(③) は後段凍結(P5-A)/後段再学習(P5-B)の2行。
- **前提**：`PYTHONPATH=./src`、上記2つの CSV。GPU 不要。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src .venv/bin/python bin/render_results_table.py \
    --compare_csv results/distill_map/final_compare3_<ts>/comparison.csv \
    --bench_csv   results/distill_map/bench_networks3_<ts>/networks.csv
  ```
- **出力先**：`results/distill_map/results_table_<YYYYMMDD_HHMMSS>/results_table.png` ＋ `results_table.csv` / `meta.json`
- **想定時間**：数秒

### scripts/compare_distillation.py — 蒸留手法の比較プロット

- **目的**：`results.csv`（method, auroc, f1, memory_mb, fps）から AUROC×メモリ散布図・指標棒グラフを描く。
- **前提**：`PYTHONPATH=./src`、対象 run ディレクトリ（`run_distillation.py` が生成）。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src python scripts/compare_distillation.py <run_dir> [--out DIR]
  ```
- **出力先**：`<run_dir>/comparison.png`（`--out` で変更可）
- **想定時間**：数秒
- **備考**：通常は `run_distillation.py` が自動で呼び出す。run_dir は timestamp 付きで `meta.json` を含むため、図は実験ごとに分離される。

### scripts/compare_continual_learning.py — 継続学習 P×R 比較プロット

- **目的**：継続学習 run 群の `results.csv` を読み、コアセット率 P × ラウンド数 R の比較図（曲線・サマリ）を出力。
- **前提**：`PYTHONPATH=./src`、`--base` 配下に `RUNS` 表のタイムスタンプ run が存在すること。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src python scripts/compare_continual_learning.py \
    [--base results/ContinualLearning] [--out DIR]
  ```
- **出力先**：`results/ContinualLearning/comparison/<YYYYMMDD_HHMMSS>/`
  - `comparison_curves.png` / `comparison_summary.png` / `meta.json` / `log.txt`
- **想定時間**：数秒

### scripts/visualize_benchmark.py — ベンチマーク結果の可視化

- **目的**：ベンチマーク結果（`Bench_*/results.csv`）から比較図（PNG / HTML）を生成。
- **前提**：`PYTHONPATH=./src`、`--results_dir` に集計済み結果。
- **コマンド例**：
  ```bash
  PYTHONPATH=./src python scripts/visualize_benchmark.py \
    [--results_dir results/Benchmark_Results] [--out DIR]
  ```
- **出力先**：`results/Benchmark_Results/viz/<YYYYMMDD_HHMMSS>/`
  - `benchmark_plot.png` / `benchmark_plot.html` / `meta.json` / `log.txt`
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

### scripts/build_inpainting_dataset.py — Inpainting ペアを MVTec 形式に再構成

- **目的**：Inpainting の `NNNNN_original.jpg` / `NNNNN_anomaly.jpg` ペアを `anomaly_dataset` と同じ MVTec 形式（`train/{good,abnormal}`・`test/{good,abnormal}`・`ground_truth/abnormal`）に再構成する。`original↔anomaly` の差分を二値化して GT マスク（白=異常, PNG 1bit）も生成する。
- **前提**：`numpy` / `Pillow` / `scipy`。展開済みの `Inpainting_test` / `Inpainting_train` ディレクトリ。
- **コマンド例**：
  ```bash
  python scripts/build_inpainting_dataset.py \
    --test-src dataset/Inpainting_test \
    --train-src dataset/Inpainting_train \
    --out dataset/inpainting_dataset \
    --threshold 30 --min-area 16
  ```
- **出力先**：`--out`（例 `dataset/inpainting_dataset/`）。test マスクは `ground_truth/abnormal/`（loader 対応）、train マスクは `ground_truth_train/abnormal/`（loader 非対象の参考用）。
- **備考**：`original` が欠損する `anomaly` はマスクを生成せず画像のみコピー。
- **想定時間**：数千枚で数分程度

### scripts/experiment_done_hook.py — 実験完了フック（自動実行）

- **目的**：実験完了時に `results.csv` のサマリーを表示する PostToolUse フック。
- **前提**：Claude Code のフックとして自動起動（手動実行は通常不要）。

### src/patchcore/runlog.py — 出力規約準拠の共通ヘルパ（ライブラリ）

- **位置づけ**：単体実行しない**ライブラリモジュール**。実験スクリプトが import して使う。
- **API**：`jst_timestamp()` / `make_run_dir(base)` / `write_meta(run_dir, extra)` / `Tee(run_dir)`（with 包囲用）/ `start_tee(run_dir)`（click 等の非包囲用、返り値を logging ハンドラに渡せば LOGGER も集約）。
- **用途**：新規スクリプトを規約（タイムスタンプ出力・`meta.json`・`log.txt`）に準拠させる際の共通基盤。

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
