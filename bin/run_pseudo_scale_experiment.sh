#!/usr/bin/env bash
# 疑似異常スケール実験（①image_to_map）：正常23,888基準で 5/10/50/100倍を比較する。
# 100倍プールを1回だけ生成・pack し、5/10/50倍は --pseudo_max で先頭サブセット参照する。
# [[pseudo-anomaly-scale-experiment]] 準拠。長時間ジョブなので実行はユーザー判断で。
#
# 使い方:
#   bash bin/run_pseudo_scale_experiment.sh <teacher_dir> [<out_root>]
# 前提: PYTHONPATH=./src / venv / GPU。保護対象(./dataset,/disk/data_area4)は読み取りのみ。
set -euo pipefail

TEACHER="${1:?teacher_dir を指定してください（正常/評価 target の precompute 用）}"
OUT_ROOT="${2:-results/distill_map/pseudo_scale}"
PY=".venv/bin/python"
export PYTHONPATH="./src"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="${OUT_ROOT}/${TS}"
mkdir -p "${RUN}"
echo "出力ルート: ${RUN}"

NORMAL_N=23888
POOL=$(( NORMAL_N * 100 ))           # 100倍プール = 2,388,800
SCALES=(5 10 50 100)

# 1) 正常/評価の Teacher target（1回）。※ train 全枚数は長時間
echo "[1/4] 正常・評価 Teacher precompute"
${PY} scripts/precompute_teacher_maps.py "${TEACHER}" \
  --gpu 0 --seed 0 --splits train test --out "${RUN}/normal_pc"
NPC="$(ls -d ${RUN}/normal_pc/*/ | tail -1)"

# 2) 100倍の blob 疑似異常を生成（1回）
echo "[2/4] blob 疑似異常を ${POOL} 枚生成"
${PY} scripts/generate_pseudo_anomalies.py \
  --num "${POOL}" --seed 0 --types blob --out "${RUN}/pseudo_gen"
BLOB="$(ls -d ${RUN}/pseudo_gen/*/ | tail -1)"

# 3) map_from_mask で packed 化（1回, Teacher 不要）
echo "[3/4] map_from_mask --pack"
${PY} scripts/precompute_teacher_maps.py \
  --map_from_mask "${BLOB}/masks" --image_dir "${BLOB}/anomaly" \
  --image_dir_split pseudo --mask_blur_sigma 2.0 --pack \
  --out "${RUN}/pseudo_pack"
PACK="$(ls -d ${RUN}/pseudo_pack/*/ | tail -1)"

# 4) 各スケールで ① 学習（他ハイパラ統一, 1:1, pseudo_max でサブセット）
echo "[4/4] 各スケール学習"
for S in "${SCALES[@]}"; do
  PMAX=$(( NORMAL_N * S ))
  echo "  === ${S}x (pseudo_max=${PMAX}) ==="
  ${PY} bin/train_distill_map.py "${NPC}" \
    --pseudo_dir "${PACK}/pseudo" --pseudo_max "${PMAX}" \
    --pattern image_to_map --name "P2_${S}x" \
    --gpu 0 --epochs 30 --batch_size 32 \
    --normal_pseudo_ratio 1.0 --steps_per_epoch 2000 \
    --lambda_map 1.0 --lambda_score 0.1 --normalize minmax \
    --out "${RUN}/train"
done

echo "完了: ${RUN}"
echo "次: eval_distill_map で各 best.ckpt を評価 → aggregate_distill_final → render_results_table（スケール別表）"
