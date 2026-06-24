#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATA_BASEDIR="${DATA_BASEDIR:-$PROJECT_ROOT/data/MMEB-V3/omniset}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_ROOT/eval_omniset_outputs/omniset}"
# Keep empty-by-default here and resolve after backbone auto-detection.
DATA_CONFIG_PATH="${DATA_CONFIG_PATH:-}"
NO_AUDIO_DATA_CONFIG_PATH="${NO_AUDIO_DATA_CONFIG_PATH:-$PROJECT_ROOT/experiments/public/eval/cross_modality_no_audio.yaml}"
OMNISET_ROTARY_DTYPE_PATCH="${OMNISET_ROTARY_DTYPE_PATCH:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-29613}"
MODEL_BACKBONE="${MODEL_BACKBONE:-auto}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
# auto | true | false
SKIP_AUDIO_TASKS="${SKIP_AUDIO_TASKS:-auto}"

# WAVE-only optional args (only effective when MODEL_BACKBONE=wave)
WAVE_PROCESSOR_PATH="${WAVE_PROCESSOR_PATH:-$MODEL_PATH}"
WAVE_TRAIN_CLASSIFY="${WAVE_TRAIN_CLASSIFY:-true}"
WAVE_CLASSIFY_TYPE="${WAVE_CLASSIFY_TYPE:-all_layer}"
WAVE_PRED_EMBEDS="${WAVE_PRED_EMBEDS:-true}"
WAVE_USE_BEATS="${WAVE_USE_BEATS:-false}"
WAVE_BEATS_PATH="${WAVE_BEATS_PATH:-}"
WAVE_BEATS_ONLY="${WAVE_BEATS_ONLY:-false}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH is required. Example: MODEL_PATH=/path/to/omni-embed-nemotron-3b bash experiments/public/eval/eval_omniset.sh"
  exit 1
fi

mkdir -p "$OUTPUT_PATH"
cd "$PROJECT_ROOT"

# Auto-detect backbone from model path when not explicitly set.
if [[ "$MODEL_BACKBONE" == "auto" ]]; then
  model_path_lower="${MODEL_PATH,,}"
  if [[ "$model_path_lower" == *"wave"* ]]; then
    EFFECTIVE_MODEL_BACKBONE="wave"
  elif [[ "$model_path_lower" == *"qwen3-vl-embedding"* || "$model_path_lower" == *"qwen3_vl"* || "$model_path_lower" == *"qwen3-vl"* ]]; then
    EFFECTIVE_MODEL_BACKBONE="qwen3_vl"
  else
    EFFECTIVE_MODEL_BACKBONE="nvomniembed"
  fi
else
  EFFECTIVE_MODEL_BACKBONE="$MODEL_BACKBONE"
fi

if [[ "$SKIP_AUDIO_TASKS" == "auto" ]]; then
  if [[ "$EFFECTIVE_MODEL_BACKBONE" == "qwen3_vl" ]]; then
    EFFECTIVE_SKIP_AUDIO_TASKS="true"
  else
    EFFECTIVE_SKIP_AUDIO_TASKS="false"
  fi
else
  EFFECTIVE_SKIP_AUDIO_TASKS="$SKIP_AUDIO_TASKS"
fi

if [[ -z "$DATA_CONFIG_PATH" ]]; then
  if [[ "$EFFECTIVE_MODEL_BACKBONE" == "qwen3_vl" && "$EFFECTIVE_SKIP_AUDIO_TASKS" == "true" ]]; then
    DATA_CONFIG_PATH="$NO_AUDIO_DATA_CONFIG_PATH"
  else
    DATA_CONFIG_PATH="$PROJECT_ROOT/experiments/public/eval/cross_modality.yaml"
  fi
fi

if [[ ! -f "$DATA_CONFIG_PATH" ]]; then
  echo "ERROR: DATA_CONFIG_PATH not found: $DATA_CONFIG_PATH"
  exit 1
fi

echo "MODEL_PATH: $MODEL_PATH"
echo "MODEL_BACKBONE: $EFFECTIVE_MODEL_BACKBONE"
echo "SKIP_AUDIO_TASKS: $EFFECTIVE_SKIP_AUDIO_TASKS"
echo "DATA_CONFIG_PATH: $DATA_CONFIG_PATH"
echo "PER_DEVICE_EVAL_BATCH_SIZE: $PER_DEVICE_EVAL_BATCH_SIZE"
echo "DATALOADER_NUM_WORKERS: $DATALOADER_NUM_WORKERS"

EVAL_ARGS=(
  --pooling "mean"
  --normalize true
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --model_backbone "$EFFECTIVE_MODEL_BACKBONE"
  --model_name "$MODEL_PATH"
  --dataset_config "$DATA_CONFIG_PATH"
  --encode_output_path "$OUTPUT_PATH"
  --data_basedir "$DATA_BASEDIR"
)

if [[ "$EFFECTIVE_MODEL_BACKBONE" == "wave" ]]; then
  EVAL_ARGS+=(
    --processor_name "$WAVE_PROCESSOR_PATH"
    --wave_train_classify "$WAVE_TRAIN_CLASSIFY"
    --wave_classify_type "$WAVE_CLASSIFY_TYPE"
    --wave_pred_embeds "$WAVE_PRED_EMBEDS"
  )
  if [[ "$WAVE_USE_BEATS" == "true" ]]; then
    if [[ -n "$WAVE_BEATS_PATH" ]]; then
      EVAL_ARGS+=(--wave_use_beats true --wave_beats_path "$WAVE_BEATS_PATH" --wave_beats_only "$WAVE_BEATS_ONLY")
    else
      echo "WARNING: WAVE_USE_BEATS=true but WAVE_BEATS_PATH is empty; fallback to --wave_use_beats false."
      EVAL_ARGS+=(--wave_use_beats false)
    fi
  else
    EVAL_ARGS+=(--wave_use_beats false)
  fi
fi

# Count visible GPUs from CUDA_VISIBLE_DEVICES (e.g., "6,7" -> 2)
NUM_GPUS=0
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
for gpu_id in "${GPU_IDS[@]}"; do
  if [[ -n "${gpu_id// }" ]]; then
    NUM_GPUS=$((NUM_GPUS + 1))
  fi
done

if [[ "$NUM_GPUS" -le 1 ]]; then
  OMNISET_ROTARY_DTYPE_PATCH="$OMNISET_ROTARY_DTYPE_PATCH" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python eval.py \
    "${EVAL_ARGS[@]}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-$NUM_GPUS}"
  OMNISET_ROTARY_DTYPE_PATCH="$OMNISET_ROTARY_DTYPE_PATCH" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_port "$MASTER_PORT" \
    --max_restarts 0 \
    eval.py \
    "${EVAL_ARGS[@]}"
fi
