#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-olm2vec}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

EMBEDDING_NPZ="${EMBEDDING_NPZ:-${SCRIPT_DIR}/outputs/step1_embeddings/embeddings_and_queries.npz}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/MMEB-V3/omniset}"
CATALOG_FILE="${CATALOG_FILE:-catalog.jsonl}"

SOURCE_MODALITY="${SOURCE_MODALITY:-t}"
TARGET_MODALITY="${TARGET_MODALITY:-i}"
QUERY_INDEX="${QUERY_INDEX:-0}"
TOPK="${TOPK:-10}"
CANDIDATE_POOL="${CANDIDATE_POOL:-target_only}"

MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/huggingface/omni-embed-nemotron-3b}"
MODEL_BACKBONE="${MODEL_BACKBONE:-nvomniembed}"
DEVICE="${DEVICE:-auto}"

OUTPUT_JSON="${OUTPUT_JSON:-${SCRIPT_DIR}/outputs/retrieval_top10.json}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ./run_retrieval_top_10.sh [extra retrieval args]

Default behavior:
  - Use precomputed first query embedding (query-index 0)
  - Retrieve top 10 from target modality candidates
  - Save JSON to outputs/retrieval_top10.json

Environment variables:
  ENV_NAME=olm2vec
  CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh

  EMBEDDING_NPZ=./outputs/step1_embeddings/embeddings_and_queries.npz
  DATASET_ROOT=$PROJECT_ROOT/data/MMEB-V3/omniset
  CATALOG_FILE=catalog.jsonl

  SOURCE_MODALITY=t
  TARGET_MODALITY=i
  QUERY_INDEX=0
  TOPK=10
  CANDIDATE_POOL=target_only   # or all

  MODEL_PATH=$HOME/.cache/huggingface/omni-embed-nemotron-3b
  MODEL_BACKBONE=nvomniembed
  DEVICE=auto

  OUTPUT_JSON=./outputs/retrieval_top10.json

Examples:
  ./run_retrieval_top_10.sh

  CANDIDATE_POOL=all ./run_retrieval_top_10.sh

  ./run_retrieval_top_10.sh \
    --query-text "a cat near a bowl" \
    --instruction "Retrieve the matching image with the same semantics."
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

python "${SCRIPT_DIR}/04_lightweight_retrieval.py" \
  --embedding-npz "${EMBEDDING_NPZ}" \
  --dataset-root "${DATASET_ROOT}" \
  --catalog-file "${CATALOG_FILE}" \
  --source-modality "${SOURCE_MODALITY}" \
  --target-modality "${TARGET_MODALITY}" \
  --query-index "${QUERY_INDEX}" \
  --topk "${TOPK}" \
  --candidate-pool "${CANDIDATE_POOL}" \
  --model-path "${MODEL_PATH}" \
  --model-backbone "${MODEL_BACKBONE}" \
  --device "${DEVICE}" \
  --save-json "${OUTPUT_JSON}" \
  "$@"
