#!/bin/bash
# スケーリングベンチマーク：画像枚数とサブサンプリング率を変えて train_time / memory_mb を計測する
#
# 使い方:
#   bash bin/benchmark_scaling.sh [GPU番号]
#
# 結果: ./results/Benchmark_Results/<timestamp>/<log_group>/results.csv
#
# 実行時間の目安（環境依存）:
#   1条件あたり数秒〜数分。全条件（14条件）で30〜60分程度。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && cd .. && pwd)"
cd "$SCRIPT_DIR"

GPU="${1:-1}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

SITE_PACKAGES="$SCRIPT_DIR/.venv/lib/python3.10/site-packages"

# Python 3.10 の実バイナリを優先順位順に探す（venv シンボリックリンクは除外）
PYTHON310=""
for _candidate in \
  /usr/bin/python3.10 \
  /opt/anaconda3/envs/tf2-2/bin/python3.10 \
  /opt/anaconda3/envs/diffusers/bin/python3.10 \
  /usr/local/bin/python3.10; do
  if [ -x "$_candidate" ]; then
    PYTHON310="$_candidate"
    break
  fi
done

if [ -z "$PYTHON310" ]; then
  echo "エラー: Python 3.10 の実バイナリが見つかりません" >&2
  exit 1
fi

echo "Python: $PYTHON310"
PYTHON="env -u PYTHONHOME PYTHONPATH=$SCRIPT_DIR/src:$SITE_PACKAGES $PYTHON310"

# ベンチマーク対象
# IMAGE_COUNTS=(10 50 100 500 1000 2000 5000 10000 20000)
IMAGE_COUNTS=(20000)
PERCENTAGES=(0.1 0.01)

RESULT_BASE="$SCRIPT_DIR/results/Benchmark_Results/$TIMESTAMP"
SUMMARY_FILE="$RESULT_BASE/summary.csv"
mkdir -p "$RESULT_BASE"

# 規約: 実行ログを log.txt に保存（R5）・再現メタ情報を meta.json に保存（R4）
exec > >(tee -a "$RESULT_BASE/log.txt") 2>&1
GIT_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
cat > "$RESULT_BASE/meta.json" <<META
{
  "command": "bash bin/benchmark_scaling.sh $*",
  "git_commit": "$GIT_COMMIT",
  "timestamp_jst": "$TIMESTAMP",
  "gpu": "$GPU",
  "image_counts": "${IMAGE_COUNTS[*]}",
  "percentages": "${PERCENTAGES[*]}"
}
META

echo "max_train_images,percentage,train_time,memory_mb" > "$SUMMARY_FILE"

# TIMEOUT_SEC=14400  # 4時間
TIMEOUT_SEC=144000

for N in "${IMAGE_COUNTS[@]}"; do
  for P in "${PERCENTAGES[@]}"; do
    # パーセント表記（0.1→P10, 0.01→P01）
    P_TAG=$(echo "$P" | sed 's/0\./P/' | sed 's/\.//g')
    LOG_GROUP="Bench_N${N}_${P_TAG}"
    echo "=============================="
    echo "  N=$N  p=$P  log_group=$LOG_GROUP"
    echo "=============================="

    if timeout "$TIMEOUT_SEC" $PYTHON bin/run_patchcore.py \
      --gpu "$GPU" --seed 0 \
      --log_group "$LOG_GROUP" \
      --log_project "Benchmark_Results/$TIMESTAMP" \
      ./results \
      patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
        --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
        --anomaly_scorer_num_nn 1 --patchsize 3 \
      sampler -p "$P" approx_greedy_coreset \
      dataset -d anomaly_dataset --resize 256 --imagesize 224 \
        --max_train_images "$N" \
        mvtec ./dataset 2>&1; then

      # results.csv から train_time と memory_mb を抽出してサマリーに追記
      RESULT_CSV=$(find "$RESULT_BASE/$LOG_GROUP" \
        -name "results.csv" 2>/dev/null | head -1)
      if [ -n "$RESULT_CSV" ]; then
        TRAIN_TIME=$(awk -F',' 'NR==2{print $11}' "$RESULT_CSV")
        MEMORY_MB=$(awk -F',' 'NR==2{print $13}' "$RESULT_CSV")
        echo "$N,$P,$TRAIN_TIME,$MEMORY_MB" >> "$SUMMARY_FILE"
        echo "  -> train_time=${TRAIN_TIME}s  memory=${MEMORY_MB}MB"
      fi
    else
      echo "  -> タイムアウト（${TIMEOUT_SEC}秒超過）: N=$N p=$P をスキップ"
      echo "$N,$P,TIMEOUT,TIMEOUT" >> "$SUMMARY_FILE"
    fi

    echo ""
  done
done

echo "=============================="
echo "ベンチマーク完了"
echo "結果フォルダ: $RESULT_BASE"
echo "サマリー: $SUMMARY_FILE"
echo "=============================="
cat "$SUMMARY_FILE"

# 規約: 実験結果は図を伴う（R3）。可視化を自動生成する。
echo ""
echo "[plot] 可視化を生成中..."
$PYTHON scripts/visualize_benchmark.py --results_dir "$RESULT_BASE" || \
  echo "  -> 可視化に失敗（Bench_*/results.csv が無い可能性）"
