# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

本リポジトリは Roth et al. (2021) "Towards Total Recall in Industrial Anomaly Detection" ([arXiv:2106.08265](https://arxiv.org/abs/2106.08265)) の PatchCore 実装です。ImageNet 事前学習済み Backbone からパッチ特徴量を抽出し、Coreset サンプリングで圧縮したメモリバンクと最近傍探索（faiss）で異常スコアを算出します。

---

## Claude への作業指示

### 応答とコミュニケーション

- ユーザーへの応答は**日本語**で行う
- 提案・修正案は、まず**結論を1行**で述べてから詳細を書く
- 前提が複数あり得る／指示が曖昧な場合は、勝手に進めず**先に確認する**

### 作業の進め方（ワークフロー）

- タスク開始時、自明でなければ**まず作業計画を1〜数行で提示**してから着手する
- 大きな変更は**小さなステップに分割**し、節目ごとに動作確認・コミットする
- エラーが出た場合は、**修正案を提示してからユーザーに確認**を取る（黙って試行錯誤を続けない）
- 既存のコードスタイル・命名・ディレクトリ構成に**合わせる**（独自流儀を持ち込まない）

### コミット規約

- **最小単位**でコミットする（1コミット1変更）
- メッセージは**日本語**で書く
- **コミット前に必ずBashで動作確認**を行う（構文チェック / テスト / `--help` 確認）
- 動作確認が取れたもののみコミットする（未確認のコードはコミットしない）
- **各タスクの変更が完了し動作確認が取れたら、ユーザーの指示を待たずにコミットする（ただし `push` はしない）**
- フォーマット：

```
<prefix>: <変更の概要（1行）>

<変更の詳細（任意、箇条書き可）>
```

使用する prefix：

| prefix | 用途 |
|--------|------|
| `feat` | 新機能の追加 |
| `fix` | バグ修正 |
| `refactor` | 動作を変えないリファクタリング |
| `perf` | パフォーマンス改善 |
| `test` | テストの追加・修正 |
| `docs` | ドキュメント・コメントの変更 |
| `chore` | ビルド・設定・依存関係の変更 |
| `style` | コードフォーマットの変更（動作に影響なし） |
| `log` | ログ・デバッグ出力の変更 |
| `git` | .gitignore・git設定の変更 |

### やってはいけないこと（確認なしで実行禁止）

これらは**ユーザーの明示的な確認・指示なしには絶対に実行しない**：

- `.env` / 秘密鍵 / 認証情報（APIキー・トークン・パスワード）の**閲覧・出力・コミット**
- `git push` / `git reset --hard` / `git rebase` / `git merge` / `git checkout`
- ファイル・ディレクトリ・ブランチの**永続的な削除**
- 保護対象ディレクトリ（`./dataset/`, `/disk/data_area4/`）への**書き込み**
- 長時間かかる実験・学習ジョブ（`bin/run_patchcore.py`, `bin/run_patchcore_venv.sh`, `bin/load_and_evaluate_patchcore.py` 等）の**自動実行**
- 本番環境・外部公開リソースへの破壊的操作

> 上記に該当する操作が必要になったら、**何をするかを具体的に説明してから確認を求める**こと。

### 不要なことはしない

- タスクに直接関係しないリファクタリングは行わない
- 将来の拡張を見越した抽象化は行わない
- コードにコメントを書くのは「なぜ」が非自明な場合のみ（簡潔に1行）

---

## 議事録・ロードマップ運用

毎週の MTG（個ゼミ）に関する成果物は以下の規約で配置・作成する。
ユーザーから「今週の議事録をまとめて」「ロードマップを書いて」等の依頼があった場合、必ず下記パス・フォーマットに従う。

### 配置とファイル名

- **議事録**：`meeting_logs/log/YYYYMMDD.md`（会議日。ユーザーが用意 or Claude が文字起こしから作成）
  - 文字起こし＋成果・進捗・アクションアイテムのサマリを含む
- **週次ロードマップ／報告資料**：`meeting_logs/roadmap/roadmap_YYYYMMDD.md`（Claude が作成）
  - 日付は **対応する MTG 開催日（議事録と同じ日付）** を用いる
  - 直近議事録のアクションアイテム・課題を起点に作成
  - ロードマップ冒頭に元議事録への相対リンク（`../log/YYYYMMDD.md`）を貼る
  - 成果物パスは `meeting_logs/roadmap/assets/YYYYMMDD/` 配下を基本とする

### 設計思想：ロードマップ＝報告資料

「**ロードマップを埋めるだけで、そのまま週次報告資料（スライド／レポート下書き）になる**」形式にする。

- 1 セクション = 1 スライド相当の粒度で構成する
- 各タスクに **アウトプット形式を事前に枠だけ作っておく**（週中に埋めるだけで発表資料が完成する）
  - 図：「図 T1-1：横軸 / 縦軸 / 凡例 / 保存先パス」をキャプション形式で先に書く
  - 表：「表 T1-1：カラム名」を先に書く
  - コード：PR or コミット URL を結果欄に書く
  - ログ：抜粋行数を指定
- 議事録の「アクションアイテム」「今後の目標」「フィードバック」を優先的にタスク化する。Speaker 名はそのまま担当として残す

### 必須セクション（スライド単位）

1. **優先度マップ**（コードブロックでツリー表示）
2. **今週のサマリー**（30秒読み上げ用）：結論／進捗ステータス／主要トピック／次回 MTG の意思決定事項
3. **タスク詳細**（各タスク1スライド相当）：
   - 担当・優先度・期限 / 背景・目的 / **アウトプット形式（図表のキャプション形式で予約）** / 計画チェックリスト / 結果欄 / ステータスチェックボックス
4. **数値サマリー横断表**（本週の主要数値を1表に集約）
5. **今週のスケジュール案**（曜日ごとの表）
6. **ブロッカー／未解決の設計課題**
7. **マイルストーン表**（期日 × 状態）
8. **週次報告ドラフト**（週末に埋めるテンプレート）：今週の成果（数値チェックリスト）／図表（パス＋1行説明）／直面した課題／次週個ゼミでの相談事項／翌週への申し送り

### 更新運用

- 既存ロードマップの改良依頼は、既存ファイルを**上書き更新**する（履歴は git に任せる）

---

## 実行・結果出力の規約

スクリプト・実験の実行方法と出力に関する共通ルール。新規スクリプトを追加・改修する際は必ず従う。

### 1. 実行方法は `docs/RUNBOOK.md` に集約する

- すべてのスクリプト（Python / bash）の実行方法は **`docs/RUNBOOK.md` に1行ずつ積み上げ**、常に最新に保つ。
- **新規スクリプトの追加・実行方法の変更は、同じコミットで RUNBOOK を更新**する（コード変更だけのコミットにしない）。
- 各エントリは「スクリプト / 目的 / 前提（`PYTHONPATH`・venv・GPU 等） / コマンド例 / 出力先 / 想定実行時間」を記載する。

### 2. 結果はタイムスタンプ付きで出力し、上書きしない

- 結果（CSV・図・モデル・ログ等）は **タイムスタンプ付きフォルダ／ファイル**に出力し、既存の実行結果を**上書きしない**。
- タイムスタンプ書式は **`YYYYMMDD_HHMMSS`（JST）** に統一する。例：`results/<project>/<run>_20260622_143000/`。
- 最新実行を参照しやすいよう、必要に応じて同階層に `latest` シンボリックリンクを貼ってよい（任意）。

### 3. 実験結果は必ず図・グラフを伴う

- 実験系スクリプトの出力には**画像／グラフを必ず含める**（数値 CSV だけで終わらせない）。
- 図の体裁は既存慣習に合わせる：`matplotlib.use("Agg")` ＋ `font.family="Noto Sans CJK JP"`、`dpi=150`、**軸ラベル・凡例・単位を必須**とする。

### 4. 再現メタ情報を同梱する（ルール A）

- 各結果フォルダに **`meta.json`** を必ず残す。最低限の項目：
  - 実行コマンド（フルコマンドライン） / **git commit hash** / `seed` / 主要ハイパーパラメータ / 環境（`torch` / `faiss` / CUDA のバージョン）/ 実行開始・終了時刻（JST）。
- 「いつの実行か」だけでなく「**どの条件・どのコードの実行か**」を後から完全に再現できる状態を担保する。

### 5. 実行ログを保存する（ルール B）

- 標準出力・標準エラーを結果フォルダ内の **`log.txt`** に保存する（`tee` 等）。失敗時の後追い解析を可能にする。

### 6. 出力先は `./results/` 配下に限定する（ルール D）

- スクリプトの出力先は **`./results/` 配下、または明示指定したパスのみ**。
- 保護対象（`./dataset/`, `/disk/data_area4/`）には**書き込まない**（読み取り専用として扱う）。

### 7. スクリプトは引数化する（ルール E）

- 出力先・`seed`・入力パス等を**ハードコードせず `argparse` で受け取る**（`sys.argv` の直読みは避ける）。
- これによりタイムスタンプ出力（ルール2）・再現（ルール4）・出力先制約（ルール6）を満たしやすくする。

---

## 環境セットアップ

```bash
# venv を使った実行（推奨）
source .venv/bin/activate

# 環境確認
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), '/ devices:', torch.cuda.device_count())"
.venv/bin/python -c "import faiss; print('faiss OK, GPU support:', hasattr(faiss, 'StandardGpuResources'))"
```

## 動作確認・テスト・静的チェック

コミット前の「動作確認」では、最低限いずれかを実行する。

```bash
# 構文チェック
python -m py_compile <file>

# テスト
PYTHONPATH=./src python3 -m pytest -v
# または
./local_run_test.sh
```

---

## パイプライン全体の流れ

### 1. 学習（PatchCore モデルの構築）

```bash
# venv スクリプト経由（推奨）
bash bin/run_patchcore_venv.sh

# 直接実行
PYTHONPATH=./src .venv/bin/python bin/run_patchcore.py \
  --gpu 1 --seed 0 --save_patchcore_model \
  --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 \
  --log_project MyDataset_Results \
  ./results \
  patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 --patchsize 3 \
  sampler -p 0.1 approx_greedy_coreset \
  dataset -d anomaly_dataset --resize 256 --imagesize 224 \
    mvtec ./dataset
```

### 2. 評価（学習済みモデルの評価）

```bash
PYTHONPATH=./src .venv/bin/python bin/load_and_evaluate_patchcore.py \
  --gpu 0 --seed 0 <save_folder> \
  patch_core_loader -p <model_path> --faiss_on_gpu \
  dataset -d anomaly_dataset --resize 256 --imagesize 224 \
    mvtec ./dataset
```

### 結果ファイル

実験結果は `./results/<log_project>/<log_group>/results.csv` に保存される：

| カラム | 説明 |
|--------|------|
| `instance_auroc` | 画像レベル異常検知 AUROC |
| `full_pixel_auroc` | ピクセルレベル異常局所化 AUROC（全ピクセル） |
| `anomaly_pixel_auroc` | ピクセルレベル異常局所化 AUROC（異常ピクセルのみ） |
| `instance_f1` | 画像レベル F1 スコア |
| `train_time` | 学習時間（秒） |
| `fps` | 推論速度 |

---

## コードアーキテクチャ

### `src/patchcore/`

- **`patchcore.py`** : メインの PatchCore クラス。特徴抽出 → パッチ埋め込み → メモリバンク構築 → 異常スコアリングを管理。
- **`backbones.py`** : WideResNet50 等の事前学習済み Backbone。`-b` オプションで指定するモデル名に対応。
- **`common.py`** : NetworkFeatureAggregator。指定レイヤ（`-le` オプション）からパッチ特徴量を抽出する。
- **`sampler.py`** : Coreset サンプリング実装（approx_greedy_coreset）。`-p` オプションでサブサンプリング率を指定。
- **`metrics.py`** : AUROC・AUPRC・F1 等の評価指標計算。
- **`datasets/mvtec.py`** : MVTec AD 形式のデータセットローダー。カスタムデータセットを追加する場合はここを参考にする。
- **`utils.py`** : ストレージ管理・デバイス設定等のユーティリティ。

### `bin/`

- **`run_patchcore.py`** : 学習エントリポイント。click の chain group で `patch_core`・`sampler`・`dataset` サブコマンドを連結する。
- **`run_patchcore_venv.sh`** : venv 環境でコンテナなしに実行するシェルスクリプト。
- **`load_and_evaluate_patchcore.py`** : 保存済みモデルをロードして評価するエントリポイント。

### `scripts/`

- **`extract_rar.py`** : RAR 形式の異常データセットを解凍するスクリプト。
- **`experiment_done_hook.py`** : 実験完了時に results.csv のサマリーを表示する PostToolUse フック（自動実行）。

### 設定システム

- CLI は **click の chain group** 方式。`run_patchcore.py` は `patch_core` / `sampler` / `dataset` の 3 サブコマンドを **この順に連結**して 1 コマンドで指定する（順序を変えると引数解釈が壊れる）。
- 各サブコマンドは独立した click オプション群を持つ（例：`patch_core` の `-b/-le/--patchsize`、`sampler` の `-p`、`dataset` の `--resize/--imagesize`）。
- 既存実験設定は `sample_training.sh` / `sample_evaluation.sh` を出発点として複製・改変するのが安全。
- ロガー類（W&B/TensorBoard）は未使用。結果は `results.csv` への追記のみ。

---

## ハマりどころ / 既知の落とし穴

- `bin/run_patchcore.py` の click サブコマンドは **連結順固定**。`patch_core → sampler → dataset` の順を崩すと無言で別解釈されることがある。
- `--faiss_on_gpu` は `faiss.StandardGpuResources` が存在する faiss-gpu ビルドでのみ有効。CPU 版 faiss だと実行時エラー。
- `PYTHONPATH=./src` を付け忘れると `import patchcore` が解決されない。`bin/` 配下の直接実行時は必須。
- 異常データは `/disk/data_area4/t-hori/2025/patchcore/anomaly_dataset/` に置かれており、ローカル `./dataset/` とは別。どちらを指しているかを混同しない。
- 継続学習で RSS が累積する不具合はコミット `bf51aad` で対処済み（中間配列の解放）。再発時は `git show bf51aad` で経緯を確認する。

---

## データ / 設定ファイルの配置

```
./dataset/                                       ローカルの学習・評価データ（保護対象）
./results/                                       実験結果・学習済みモデル
/disk/data_area4/t-hori/2025/patchcore/          外部ストレージのデータ（保護対象）
  anomaly_dataset/
    ground_truth/abnormal/                       異常画像のグラウンドトゥルース
meeting_logs/
  log/YYYYMMDD.md                                MTG 議事録
  roadmap/roadmap_YYYYMMDD.md                    週次ロードマップ／報告資料
  roadmap/assets/YYYYMMDD/                       ロードマップから参照する図表等の成果物
```

MVTec AD 形式のディレクトリ構造：

```
./dataset/<カテゴリ>/
  train/good/            正常サンプル（学習用）
  test/good/             正常サンプル（評価用）
  test/<欠陥名>/         異常サンプル（評価用）
  ground_truth/<欠陥名>/ マスク画像（PNG、異常ピクセルが白）
```

---

## このファイル（CLAUDE.md）の育て方

- 作業中に**新しい教訓・ハマりどころ・運用ルール**が見つかったら、該当セクションに**簡潔に追記**する
- 追記は「なぜそうするか」が非自明な場合に**理由を1行**添える
- 肥大化したら使われていない記述を削る。**短く正確**を優先する
