#!/usr/bin/env bash
# 疑似異常スケール実験④（A=マスクターゲット, ①image_to_map）：
# 正常のみ P1 と、正常+疑似 P2 を 5/10/50/100倍で学習→評価する。
# 100倍packから --pseudo_max でサブセット参照（再生成なし）。他ハイパラ統一・1:1。
# [[pseudo-anomaly-scale-experiment]]。長時間GPUジョブ。
#
# 使い方: bash bin/run_scale_train_A.sh <pack_pseudo_dir> [<epochs>] [<steps_per_epoch>]
set -euo pipefail

PACK="${1:?pack の pseudo ディレクトリ（.../<ts>/pseudo）を指定}"
EPOCHS="${2:-30}"
STEPS="${3:-2000}"
SCALES_ARG="${4:-5 10 50 100}"   # 例: "5 10" で10倍まで
NPC="results/distill_map/precomputed/20260622_222843_all"   # 再利用（round_21, train+test）
PY=".venv/bin/python"; export PYTHONPATH="./src"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="results/distill_map/pseudo_scale_A/${TS}"
mkdir -p "$OUT"
NORMAL_N=23888
COMMON="--pattern image_to_map --gpu 0 --epochs ${EPOCHS} --batch_size 32 \
  --steps_per_epoch ${STEPS} --normal_pseudo_ratio 1.0 --cache_normal 1 \
  --lambda_map 1.0 --lambda_score 0.1 --normalize minmax --num_workers 8"

train_eval () {  # name pseudo_args...
  local name="$1"; shift
  echo "=== TRAIN ${name} ==="
  ${PY} bin/train_distill_map.py "$NPC" --name "$name" $COMMON "$@" --out "$OUT"
  local run; run="$(ls -dt ${OUT}/${name}_*/ | head -1)"
  echo "=== EVAL ${name} ==="
  ${PY} bin/eval_distill_map.py "$NPC" --checkpoint "${run}checkpoints/best.ckpt" \
    --name "eval_${name}" --gpu 0 --num_vis 4 --out "$OUT"
}

# P1: 正常のみ（スケール非依存ベースライン）
train_eval "P1_normal_only"
# P2: 正常+疑似。サブセットは stride（全ソースを均等に含む）
for S in ${SCALES_ARG}; do
  train_eval "P2_${S}x" --pseudo_dir "$PACK" --pseudo_max $(( NORMAL_N * S )) \
    --pseudo_subset stride
done

echo "完了: ${OUT}"
echo "次: bin/render_results_table 相当でスケール別表＋倍率×AUROC図を作成"
