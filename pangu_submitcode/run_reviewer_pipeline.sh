#!/usr/bin/env bash
set -euo pipefail

# Hard-mode pipeline aligned to paper sections:
#   Section 4.2 -> stage1
#   Section 4.3 -> stage2
#   Section 4.4 -> stage3

if [[ $# -lt 1 ]]; then
  echo "usage: bash pangu_submitcode/run_reviewer_pipeline.sh <stage1|stage2|stage3>"
  exit 1
fi

stage="$1"

if [[ "$stage" == "stage1" ]]; then
  echo "[stage1] Section 4.2 Global Weather Model Prediction"
  echo "Requires manual date edits in: data_prepare.py / inference.py / forecast_decode.py"
  python pangu_submitcode/data_prepare.py
  python pangu_submitcode/inference.py
  python pangu_submitcode/forecast_decode.py
  exit 0
fi

if [[ "$stage" == "stage2" ]]; then
  echo "[stage2] Section 4.3 Physics Distillation Module"
  : "${DATASET_TAG:?DATASET_TAG is required}"
  : "${UPPER_RAW:?UPPER_RAW is required}"
  : "${SURFACE_RAW:?SURFACE_RAW is required}"
  : "${TIME_LABELS:?TIME_LABELS is required}"
  : "${PACK_OUTPUT_DIR:?PACK_OUTPUT_DIR is required}"
  python pangu_submitcode/pdm_manual_pack.py \
    --dataset_tag "$DATASET_TAG" \
    --upper_raw "$UPPER_RAW" \
    --surface_raw "$SURFACE_RAW" \
    --time_labels "$TIME_LABELS" \
    --output_dir "$PACK_OUTPUT_DIR"
  exit 0
fi

if [[ "$stage" == "stage3" ]]; then
  echo "[stage3] Section 4.4 GRCMI-WPF Training"
  : "${TRAIN_CONFIG:?TRAIN_CONFIG is required}"
  : "${TRAIN_OUTPUT_ROOT:?TRAIN_OUTPUT_ROOT is required}"
  python pangu_submitcode/train_grcmi_wpf.py \
    --config "$TRAIN_CONFIG" \
    --output_root "$TRAIN_OUTPUT_ROOT"
  exit 0
fi

echo "unknown stage: $stage"
exit 1
