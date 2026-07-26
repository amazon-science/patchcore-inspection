#!/usr/bin/env bash
# 定量評価表の「画像→特徴map」行（＝③前段のみ蒸留）を B方式10x 条件で埋める。
#
# この行の Student は異常マップを出さないため、後段は **Teacher の faiss メモリバンクを
# そのまま使う**（画像 →[前段 ResNet18-FPN]→ F_S →[bank NN]→ 異常マップ/スコア）。
# 議事録 20260707 で先生Bが挙げた「ResNet50 でバンクを作り 18 で見比べる素直な構成」。
#
# run_pseudoB_23.sh とは別ファイルにしてある：実行中の bash スクリプトを編集すると
# バイトオフセットのズレで後続コマンドが壊れるため、追記ではなく独立スクリプトにする。
# 前提：run_pseudoB_23.sh のステップ[2]（③前段 P3B_front）が完了していること。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=./src
PY=.venv/bin/python

GPU="${GPU:-1}"
OUT="${OUT:-results/distill_map/pseudoB_10x}"
TFEAT="${TFEAT:-/tmp/t-hori/bfai/allfeat_local/20260623_042006}"
TEACHER="${TEACHER:-results/ContinualLearning/20260526_比較検討/20260527_064537_N1000_P5/round_21}"

FRONT="${FRONT:-}"
if [[ -z "$FRONT" ]]; then
  RF=$(ls -1dt "$OUT"/P3B_front_*/ 2>/dev/null | head -1)
  [[ -n "$RF" ]] || { echo "ERROR: ③前段 (P3B_front_*) が見つかりません。先に run_pseudoB_23.sh の[2]完了を待つこと"; exit 1; }
  FRONT="${RF}checkpoints/best.ckpt"
fi
[[ -f "$FRONT" ]] || { echo "ERROR: best.ckpt がありません: $FRONT"; exit 1; }
echo "前段: $FRONT"

# faiss は src/patchcore/common.py で dev0 固定のため、CUDA_VISIBLE_DEVICES で
# 使いたい GPU を論理0に見せてから --gpu 0 を渡す（--gpu 1 だけでは GPU0 を掴んで OOM）。
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=./src $PY bin/eval_front_with_bank.py "$TFEAT" \
  --front_checkpoint "$FRONT" --teacher "$TEACHER" \
  --name eval_front_bank_10x --gpu 0 --batch_size 8 --num_workers 4 \
  --out "$OUT"

echo "完了 $(date +%H:%M:%S)  → $OUT/eval_front_bank_10x_*/metrics/front_bank_metrics.json"
