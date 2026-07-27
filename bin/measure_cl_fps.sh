#!/bin/bash
# 継続学習9条件（N x P）の最終ラウンドモデルを再評価し、fps を計測する。
#
# 使い方:
#   bash bin/measure_cl_fps.sh                 # 既定: GPU=1, fpsのみ（案B, 約30分）
#   GPU=1 WITH_PIXEL=1 bash bin/measure_cl_fps.sh   # ピクセルAUROCも計算（案C, 約1時間）
#
# 結果: ./results/ContinualLearning/fps_<timestamp>/<N####_P##>/results.csv
#       ./results/ContinualLearning/fps_<timestamp>/summary.csv
#       ./results/ContinualLearning/fps_<timestamp>/fps_summary.png

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && cd .. && pwd)"
cd "$SCRIPT_DIR"

GPU="${GPU:-1}"
WITH_PIXEL="${WITH_PIXEL:-0}"
SRC_BASE="${SRC_BASE:-results/ContinualLearning/20260526_比較検討}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULT_BASE="$SCRIPT_DIR/results/ContinualLearning/fps_$TIMESTAMP"

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "エラー: $PYTHON が見つかりません" >&2
  exit 1
fi

mkdir -p "$RESULT_BASE"

# 規約: 実行ログを log.txt に保存（R5）
exec > >(tee -a "$RESULT_BASE/log.txt") 2>&1

PIXEL_FLAG="--skip_pixel_metrics"
[ "$WITH_PIXEL" = "1" ] && PIXEL_FLAG=""

echo "=============================="
echo "  継続学習9条件の fps 計測"
echo "  GPU            : $GPU"
echo "  pixel metrics  : $([ "$WITH_PIXEL" = "1" ] && echo "計算する" || echo "スキップ")"
echo "  入力           : $SRC_BASE"
echo "  出力           : $RESULT_BASE"
echo "=============================="

echo "condition,n_per_round,coreset_pct,round,memory_bank_size,instance_auroc,instance_f1,fps,full_pixel_auroc,anomaly_pixel_auroc" \
  > "$RESULT_BASE/summary.csv"

for RUN_DIR in "$SRC_BASE"/*_N*_P*/; do
  RUN_DIR="${RUN_DIR%/}"
  RUN_NAME=$(basename "$RUN_DIR")

  # ディレクトリ名 <TS>_N<枚数>_P<コアセット%> から条件を取り出す
  N=$(echo "$RUN_NAME" | sed -n 's/.*_N\([0-9]*\)_P.*/\1/p')
  P=$(echo "$RUN_NAME" | sed -n 's/.*_P\([0-9]*\)$/\1/p')
  LABEL="N${N}_P${P}"

  # 最終ラウンドのモデルを使う
  LAST_ROUND=$(ls -d "$RUN_DIR"/round_* 2>/dev/null | sort -V | tail -1)
  if [ -z "$LAST_ROUND" ] || [ ! -f "$LAST_ROUND/nnscorer_search_index.faiss" ]; then
    echo "  -> スキップ: $RUN_NAME に最終ラウンドのモデルがありません"
    continue
  fi
  ROUND_NO=$(basename "$LAST_ROUND" | sed 's/round_//')

  OUT_DIR="$RESULT_BASE/$LABEL"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "=============================="
  echo "  $LABEL  (round=$ROUND_NO)"
  echo "  model: $LAST_ROUND"
  echo "=============================="

  # faiss の GPU インデックスは common.py で device 0 固定のため、
  # CUDA_VISIBLE_DEVICES で使いたい GPU を論理0に見せてから --gpu 0 を渡す。
  # （--gpu 1 だけでは faiss が GPU0 を掴んで OOM する）
  if ! CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=./src $PYTHON \
      bin/load_and_evaluate_patchcore.py \
      --gpu 0 --seed 0 $PIXEL_FLAG "$OUT_DIR" \
      patch_core_loader -p "$LAST_ROUND" --faiss_on_gpu \
      dataset -d anomaly_dataset --resize 256 --imagesize 224 \
        mvtec ./dataset; then
    echo "  -> 失敗: $LABEL"
    continue
  fi

  BANK=$(awk -F',' -v r="$ROUND_NO" '$1==r{print $5}' "$RUN_DIR/results.csv")
  $PYTHON - "$OUT_DIR/results.csv" "$LABEL" "$N" "$P" "$ROUND_NO" "$BANK" \
    >> "$RESULT_BASE/summary.csv" <<'PY'
import csv, sys
csv_path, label, n, p, rnd, bank = sys.argv[1:7]
row = list(csv.DictReader(open(csv_path)))[-1]
def g(k):
    v = row.get(k, "")
    return v if v not in (None, "") else "nan"
print(",".join([label, n, p, rnd, bank, g("instance_auroc"), g("instance_f1"),
                g("fps"), g("full_pixel_auroc"), g("anomaly_pixel_auroc")]))
PY
done

echo ""
echo "=============================="
echo "計測完了"
echo "サマリー: $RESULT_BASE/summary.csv"
echo "=============================="
cat "$RESULT_BASE/summary.csv"

# 規約: 実験結果は図を伴う（R3）
echo ""
echo "[plot] 可視化を生成中..."
PYTHONPATH=./src $PYTHON scripts/plot_cl_fps.py --summary "$RESULT_BASE/summary.csv" \
  --out "$RESULT_BASE/fps_summary.png" || echo "  -> 可視化に失敗"
