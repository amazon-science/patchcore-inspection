#!/usr/bin/env bash
# =============================================================================
# run_final_distill_comparison.sh
#   マップ蒸留の最終比較実験（2x2）を train -> eval まで一括実行する。
#
#   比較条件（差分は --pattern と --pseudo_dir の2軸のみ。他は完全統一）:
#     P1: image_to_map   / 正常のみ
#     P2: image_to_map   / 正常 + 疑似異常（公式 inpainting_dataset 由来）
#     P3: feature_to_map / 正常のみ
#     P4: feature_to_map / 正常 + 疑似異常
#
#   データセット（全条件共通）:
#     Teacher  : results/ContinualLearning/.../round_21（PatchCore 1000x5% round21）
#     正常/test: dataset/anomaly_dataset（train/good 23,888 / test 1,388[異常116]）
#     疑似異常 : dataset/inpainting_dataset/train/abnormal（公式・P2/P4のみ）
#
#   GPU 割当: GPU_A=image系(P1->P2 直列) / GPU_B=feature系(P3->P4 直列) を並行。
#
#   注意: 長時間ジョブ。手動実行のみ（自動実行しない）。
# =============================================================================
set -euo pipefail

# --- リポジトリルートへ移動 ---------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- 設定（環境変数で上書き可） ----------------------------------------------
PY="${PY:-.venv/bin/python}"
ALLFEAT="${ALLFEAT:-results/distill_map/precomputed_allfeat/20260623_042006}"
PSEUDO="${PSEUDO:-results/distill_map/precomputed_pseudo_official/20260623_055836/pseudo}"
TEACHER="${TEACHER:-results/ContinualLearning/20260526_比較検討/20260527_064537_N1000_P5/round_21}"
GPU_A="${GPU_A:-0}"   # image_to_map 系
GPU_B="${GPU_B:-1}"   # feature_to_map 系

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH="${BATCH:-32}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_VIS="${NUM_VIS:-8}"

OUT_ROOT="${OUT_ROOT:-results/distill_map}"

# --- 統一ハイパラ（全4条件で共通） -------------------------------------------
COMMON=(
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --early_stopping 1
  --patience "${PATIENCE}"
  --batch_size "${BATCH}"
  --learning_rate "${LR}"
  --normalize minmax
  --map_loss smooth_l1
  --lambda_map 1.0
  --lambda_score 0.1
  --use_weighted_map_loss 1
  --anomaly_weight_alpha 5.0
  --score_pooling max
  --fpn_out_channels 128
  --decoder_channels 128
  --pretrained 1
  --freeze_backbone 0
  --num_workers "${NUM_WORKERS}"
)

# --- マスタ出力フォルダ（タイムスタンプ・上書きしない） ----------------------
TS="$(TZ=Asia/Tokyo date +%Y%m%d_%H%M%S)"
MASTER="${OUT_ROOT}/FINAL_comparison_${TS}"
mkdir -p "${MASTER}"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "=== FINAL distill comparison ===" | tee "${MASTER}/orchestrator.log"
{
  echo "master_dir : ${MASTER}"
  echo "git_commit : ${GIT_COMMIT}"
  echo "allfeat    : ${ALLFEAT}"
  echo "pseudo     : ${PSEUDO}"
  echo "teacher    : ${TEACHER}"
  echo "gpu_A/B    : ${GPU_A} / ${GPU_B}"
  echo "epochs     : ${EPOCHS} (early_stopping patience=${PATIENCE})"
} | tee -a "${MASTER}/orchestrator.log"

# --- 前提チェック ------------------------------------------------------------
for d in "${ALLFEAT}/train" "${ALLFEAT}/test" "${PSEUDO}" "${TEACHER}"; do
  if [[ ! -e "${d}" ]]; then
    echo "ERROR: 必要なパスが見つかりません: ${d}" | tee -a "${MASTER}/orchestrator.log"
    exit 1
  fi
done

# --- 1 条件分の train -> eval を実行する関数 ---------------------------------
#   引数: <name> <pattern> <use_pseudo:0|1> <gpu>
run_one() {
  local name="$1" pattern="$2" use_pseudo="$3" gpu="$4"
  local lane_log="${MASTER}/${name}.log"

  echo "[${name}] TRAIN start (pattern=${pattern}, pseudo=${use_pseudo}, gpu=${gpu})" \
    | tee -a "${MASTER}/orchestrator.log"

  local pseudo_args=()
  if [[ "${use_pseudo}" == "1" ]]; then
    pseudo_args=(--pseudo_dir "${PSEUDO}")
  fi

  PYTHONPATH=./src "${PY}" bin/train_distill_map.py "${ALLFEAT}" \
    --name "${name}" --pattern "${pattern}" --gpu "${gpu}" \
    "${pseudo_args[@]}" "${COMMON[@]}" >>"${lane_log}" 2>&1

  # 学習出力フォルダ（name 接頭辞は一意）の最新を特定
  local run_dir
  run_dir="$(ls -dt "${OUT_ROOT}/${name}"_*/ 2>/dev/null | head -1 || true)"
  if [[ -z "${run_dir}" || ! -f "${run_dir}checkpoints/best.ckpt" ]]; then
    echo "[${name}] ERROR: best.ckpt が見つかりません (run_dir=${run_dir})" \
      | tee -a "${MASTER}/orchestrator.log"
    return 1
  fi
  echo "[${name}] train_dir = ${run_dir}" | tee -a "${MASTER}/orchestrator.log"

  echo "[${name}] EVAL start" | tee -a "${MASTER}/orchestrator.log"
  PYTHONPATH=./src "${PY}" bin/eval_distill_map.py "${ALLFEAT}" \
    --checkpoint "${run_dir}checkpoints/best.ckpt" \
    --name "eval_${name}" --gpu "${gpu}" \
    --num_vis "${NUM_VIS}" --teacher "${TEACHER}" >>"${lane_log}" 2>&1

  local eval_dir
  eval_dir="$(ls -dt "${OUT_ROOT}/eval_${name}"_*/ 2>/dev/null | head -1 || true)"
  echo "[${name}] eval_dir  = ${eval_dir}" | tee -a "${MASTER}/orchestrator.log"
  # 後段の集計用にパスを記録
  echo "${name}|${pattern}|${use_pseudo}|${run_dir}|${eval_dir}" >>"${MASTER}/manifest.txt"
  echo "[${name}] DONE" | tee -a "${MASTER}/orchestrator.log"
}

# --- GPU レーン（同一 GPU 内は直列、レーン間は並行） -------------------------
lane_A() {
  run_one "FINAL_P1_image_normal"   image_to_map   0 "${GPU_A}"
  run_one "FINAL_P2_image_pseudo"   image_to_map   1 "${GPU_A}"
}
lane_B() {
  run_one "FINAL_P3_feature_normal" feature_to_map 0 "${GPU_B}"
  run_one "FINAL_P4_feature_pseudo" feature_to_map 1 "${GPU_B}"
}

: >"${MASTER}/manifest.txt"
lane_A & PID_A=$!
lane_B & PID_B=$!

FAIL=0
wait "${PID_A}" || FAIL=1
wait "${PID_B}" || FAIL=1
if [[ "${FAIL}" -ne 0 ]]; then
  echo "ERROR: いずれかのレーンが失敗しました。${MASTER}/*.log を確認してください。" \
    | tee -a "${MASTER}/orchestrator.log"
  exit 1
fi

# --- 横断集計（image 中心 + 効率）: summary.csv + 図 -------------------------
echo "=== 集計 ===" | tee -a "${MASTER}/orchestrator.log"
PYTHONPATH=./src "${PY}" - "${MASTER}" <<'PYEOF' 2>&1 | tee -a "${MASTER}/orchestrator.log"
import json, os, sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"

master = sys.argv[1]
manifest = os.path.join(master, "manifest.txt")
rows = []
teacher_row = None
with open(manifest) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        name, pattern, use_pseudo, run_dir, eval_dir = line.split("|")
        im = json.load(open(os.path.join(eval_dir, "metrics", "image_metrics.json")))
        eff_path = os.path.join(eval_dir, "metrics", "efficiency_metrics.json")
        eff = json.load(open(eff_path)) if os.path.exists(eff_path) else {}
        fid_path = os.path.join(eval_dir, "metrics", "distillation_fidelity.json")
        fid = json.load(open(fid_path)) if os.path.exists(fid_path) else {}
        s = im.get("student", {})
        se = eff.get("student", {})
        rows.append({
            "id": name.replace("FINAL_", "").split("_")[0],
            "name": name,
            "pattern": pattern,
            "pseudo": "yes" if use_pseudo == "1" else "no",
            "auroc": s.get("auroc"),
            "ap": s.get("average_precision"),
            "f1_max": s.get("f1_max"),
            "map_mse": fid.get("map_mse"),
            "map_mae": fid.get("map_mae"),
            "fps": se.get("fps"),
            "infer_ms": se.get("inference_time_ms"),
            "params_M": (se.get("parameters") or 0) / 1e6 if se.get("parameters") else None,
            "model_size_MB": se.get("model_size_mb"),
            "gpu_mem_MB": se.get("gpu_memory_mb") or se.get("gpu_memory"),
        })
        if teacher_row is None and "teacher" in im:
            t = im["teacher"]; te = eff.get("teacher", {})
            teacher_row = {
                "id": "T0", "name": "PatchCore_Teacher", "pattern": "-", "pseudo": "-",
                "auroc": t.get("auroc"), "ap": t.get("average_precision"),
                "f1_max": t.get("f1_max"),
                "map_mse": 0.0, "map_mae": 0.0,
                "fps": te.get("fps"),
                "infer_ms": te.get("inference_time_ms"),
                "params_M": (te.get("parameters") or 0) / 1e6 if te.get("parameters") else None,
                "model_size_MB": te.get("model_size_mb"),
                "gpu_mem_MB": te.get("gpu_memory_mb") or te.get("gpu_memory"),
            }

order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
rows.sort(key=lambda r: order.get(r["id"], 9))
all_rows = ([teacher_row] if teacher_row else []) + rows

cols = ["id", "name", "pattern", "pseudo", "auroc", "ap", "f1_max",
        "map_mse", "map_mae",
        "fps", "infer_ms", "params_M", "model_size_MB", "gpu_mem_MB"]
csv_path = os.path.join(master, "summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in all_rows:
        w.writerow({k: r.get(k) for k in cols})
print("wrote", csv_path)

def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else ("-" if v is None else str(v))
print(f"{'id':<4}{'pattern':<16}{'pseudo':<8}{'AUROC':<9}{'AP':<9}{'F1max':<9}{'FPS':<9}")
for r in all_rows:
    print(f"{r['id']:<4}{r['pattern']:<16}{r['pseudo']:<8}"
          f"{fmt(r['auroc']):<9}{fmt(r['ap']):<9}{fmt(r['f1_max']):<9}{fmt(r['fps']):<9}")

# --- 図: 4条件の AUROC 棒グラフ（Teacher を破線で基準表示） ---
labels = [r["id"] for r in rows]
aurocs = [r["auroc"] or 0 for r in rows]
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(labels, aurocs, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
if teacher_row and teacher_row["auroc"]:
    ax.axhline(teacher_row["auroc"], ls="--", color="gray",
               label=f"Teacher AUROC={teacher_row['auroc']:.3f}")
    ax.legend()
for b, v in zip(bars, aurocs):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Instance AUROC")
ax.set_xlabel("条件（P1:画像/正常, P2:画像/疑似, P3:特徴/正常, P4:特徴/疑似）")
ax.set_title("マップ蒸留 最終比較: Student instance AUROC")
fig.tight_layout()
fig_path = os.path.join(master, "summary_auroc.png")
fig.savefig(fig_path, dpi=150)
print("wrote", fig_path)
PYEOF

echo "=== 完了: 集計は ${MASTER}/summary.csv / summary_auroc.png ===" \
  | tee -a "${MASTER}/orchestrator.log"
