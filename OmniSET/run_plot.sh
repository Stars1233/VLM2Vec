#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Environment settings
ENV_NAME="${ENV_NAME:-olm2vec}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

# Pipeline settings
DATASET_ROOT="${DATASET_ROOT:-${HOME}/.cache/huggingface/datasets/MMEB-V3/mscoco-omni}"
META_FILE="${META_FILE:-${DATASET_ROOT}/mscoco_cmret_all.jsonl}"
CATALOG_FILE="${CATALOG_FILE:-${DATASET_ROOT}/catalog.jsonl}"
MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/huggingface/omni-embed-nemotron-3b}"
MODEL_BACKBONE="${MODEL_BACKBONE:-nvomniembed}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DEVICE="${DEVICE:-auto}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"
STEP1_DIR="${STEP1_DIR:-${OUTPUT_ROOT}/step1_embeddings}"
STEP2_DIR="${STEP2_DIR:-${OUTPUT_ROOT}/step2_analysis}"
STEP3_DIR="${STEP3_DIR:-${OUTPUT_ROOT}/step3_figures}"

EMBEDDING_NPZ="${EMBEDDING_NPZ:-${STEP1_DIR}/embeddings_and_queries.npz}"
FORCE_REEMBED="${FORCE_REEMBED:-0}"

TSNE_PERPLEXITY="${TSNE_PERPLEXITY:-30}"
TSNE_METRIC="${TSNE_METRIC:-cosine}"
TSNE_RANDOM_STATE="${TSNE_RANDOM_STATE:-42}"
QUERY_SHIFT_PROJ_K="${QUERY_SHIFT_PROJ_K:-24}"
CLUSTER_SAMPLE_RATIO="${CLUSTER_SAMPLE_RATIO:-0.25}"
DPI="${DPI:-300}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ./run_plot.sh

This script runs OmniSET plotting end-to-end:
  - If embeddings exist, run Step2 + Step3
  - If embeddings are missing (or FORCE_REEMBED=1), run Step1 first

Optional environment variables:
  ENV_NAME=olm2vec
  CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh

  DATASET_ROOT=$HOME/.cache/huggingface/datasets/MMEB-V3/mscoco-omni
  META_FILE=$HOME/.cache/huggingface/datasets/MMEB-V3/mscoco-omni/mscoco_cmret_all.jsonl
  CATALOG_FILE=$HOME/.cache/huggingface/datasets/MMEB-V3/mscoco-omni/catalog.jsonl
  MODEL_PATH=$HOME/.cache/huggingface/omni-embed-nemotron-3b
  MODEL_BACKBONE=nvomniembed
  MAX_SAMPLES=0
  DEVICE=auto

  OUTPUT_ROOT=./outputs
  STEP1_DIR=./outputs/step1_embeddings
  STEP2_DIR=./outputs/step2_analysis
  STEP3_DIR=./outputs/step3_figures
  EMBEDDING_NPZ=./outputs/step1_embeddings/embeddings_and_queries.npz
  FORCE_REEMBED=0

  TSNE_PERPLEXITY=30
  TSNE_METRIC=cosine
  TSNE_RANDOM_STATE=42
  QUERY_SHIFT_PROJ_K=24
  CLUSTER_SAMPLE_RATIO=0.25
  DPI=300

Example:
  CUDA_VISIBLE_DEVICES=0 FORCE_REEMBED=1 ./run_plot.sh
EOF
  exit 0
fi

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[ERROR] Cannot find conda init script: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}"
fi

echo "[INFO] Using conda env: ${CONDA_DEFAULT_ENV:-unknown}"
python - <<'PY'
import torch
print(f"[INFO] torch.cuda.is_available = {torch.cuda.is_available()}")
print(f"[INFO] torch.cuda.device_count = {torch.cuda.device_count()}")
print("[INFO] Embedding script uses a single device (one GPU if CUDA is enabled).")
PY

mkdir -p "${STEP1_DIR}" "${STEP2_DIR}" "${STEP3_DIR}"

if [[ "${FORCE_REEMBED}" == "1" || ! -f "${EMBEDDING_NPZ}" ]]; then
  echo "[RUN] Step1: compute embeddings"
  python "${SCRIPT_DIR}/01_compute_embeddings.py" \
    --dataset-root "${DATASET_ROOT}" \
    --meta-file "${META_FILE}" \
    --catalog-file "${CATALOG_FILE}" \
    --model-path "${MODEL_PATH}" \
    --model-backbone "${MODEL_BACKBONE}" \
    --output-dir "${STEP1_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --device "${DEVICE}"
else
  echo "[INFO] Reusing existing embedding bundle: ${EMBEDDING_NPZ}"
fi

echo "[RUN] Step2: compute t-SNE and cosine shifts"
python "${SCRIPT_DIR}/02_compute_tsne_and_shift.py" \
  --embedding-npz "${EMBEDDING_NPZ}" \
  --output-dir "${STEP2_DIR}" \
  --tsne-perplexity "${TSNE_PERPLEXITY}" \
  --tsne-random-state "${TSNE_RANDOM_STATE}" \
  --tsne-metric "${TSNE_METRIC}" \
  --query-shift-proj-k "${QUERY_SHIFT_PROJ_K}"

echo "[RUN] Step3: plot t-SNE clusters and heatmap"
python "${SCRIPT_DIR}/03_plot_tsne_and_heatmap.py" \
  --cluster-points-csv "${STEP2_DIR}/tsne_cluster_points.csv" \
  --delta-matrix-csv "${STEP2_DIR}/delta_cosine_matrix.csv" \
  --output-dir "${STEP3_DIR}" \
  --cluster-sample-ratio "${CLUSTER_SAMPLE_RATIO}" \
  --dpi "${DPI}"

echo "[DONE] Figures are in: ${STEP3_DIR}"
