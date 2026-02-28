#!/usr/bin/env bash

echo "==> Environment"
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

# cd projects/VLM2Vec/ || exit

# ==============================================================================
# Configuration
# ==============================================================================
CUDA_VISIBLE_DEVICES="0"
BATCH_SIZE=8
# NPROC=2 多卡
# NPROC=2 
# BATCH_SIZE=8
# MODALITIES=("image" "video" "tool" "visdoc" "audio" "text")
MODALITIES=("audio")
DATA_BASEDIR=/data/mengrui/.cache/huggingface/datasets/MMEB-V3
OUTPUT_BASEDIR=exps/vlm2vec


# ==> Define models and their base output paths here
# Format: "MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH[;CHECKPOINT_PATH]"
declare -a MODEL_SPECS
# 选项1: 使用本地Qwen2-VL-2B + VLM2Vec-V2.0适配器（推荐，避免网络下载）
# MODEL_SPECS+=( "/code/.cache/huggingface/Qwen2-VL-2B;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V2.0-Qwen2VL-2B;/code/.cache/huggingface/VLM2Vec-V2.0" )
# MODEL_SPECS+=( "/code/.cache/huggingface/Qwen2.5-Omni-3B;qwen2_5_omni;$OUTPUT_BASEDIR/VLM2Vec-V3.0-Qwen2_5omni-3B;/code/OLM2VEC_and_MMEB-V3/VLM2Vec1/VLM2Vec/exps/output_model/Qwen2_5Omni_3B.audio.lora16.BS512.IB64.GCq8p8.NormTemp002.lr5e5.step5kwarm100/checkpoint-2850" )

MODEL_SPECS+=( "/data/mengrui/OLM2Vec/OLM2Vec/exps/output_model/Qwen2_5Omni_3B.audio.lora16.BS512.IB64.GCq8p8.NormTemp002.lr5e5.step5kwarm100/checkpoint-600;nvomniembed;$OUTPUT_BASEDIR/ours" )
# MODEL_SPECS+=( "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-2B-Instruct" )
# MODEL_SPECS+=( "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-7B-Instruct" )
# MODEL_SPECS+=( "code-kunkun/LamRA-Ret;lamra;$OUTPUT_BASEDIR/LamRA-Ret" )
# MODEL_SPECS+=( "vidore/colpali-v1.3;colpali;$OUTPUT_BASEDIR/colpali-v1.3" )


# ==============================================================================
# Main Execution Loop
# ==============================================================================
# Loop through each model specification
for spec in "${MODEL_SPECS[@]}"; do
  # Parse the model specification: MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH[;CHECKPOINT_PATH]
  IFS=';' read -r MODEL_NAME MODEL_BACKBONE BASE_OUTPUT_PATH CHECKPOINT_PATH <<< "$spec"

  echo "================================================="
  echo "🚀 Processing Model: $MODEL_NAME"
  echo "================================================="

  # Loop through each modality for the current model
  for MODALITY in "${MODALITIES[@]}"; do
    DATA_CONFIG_PATH="experiments/public/eval/$MODALITY.yaml"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$MODALITY/"

    echo "-------------------------------------------------"
    echo "  - Modality: $MODALITY"
    echo "  - Output Path: $OUTPUT_PATH"

    # Ensure the output directory exists
    mkdir -p "$OUTPUT_PATH"
# --pooling eos
    # cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node $NPROC eval.py \
    cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python eval.py \
      --pooling mean \
      --normalize true \
      --per_device_eval_batch_size $BATCH_SIZE \
      --model_backbone \"$MODEL_BACKBONE\" \
      --model_name \"$MODEL_NAME\" \
      --dataset_config \"$DATA_CONFIG_PATH\" \
      --encode_output_path \"$OUTPUT_PATH\" \
      --data_basedir \"$DATA_BASEDIR\" \
      --lora true \
      --processor_name /code/.cache/huggingface/Qwen2.5-Omni-3B "
    # Add checkpoint_path if specified new added --lora true：--lora true \
      # --processor_name /code/.cache/huggingface/Qwen2.5-Omni-3B
    if [ -n "$CHECKPOINT_PATH" ]; then
      cmd="$cmd --checkpoint_path \"$CHECKPOINT_PATH\""
    fi
    
    echo "  - Executing command..."
    # echo "$cmd" # Uncomment for debugging the exact command
    eval "$cmd"
    echo "  - Done."
    echo "-------------------------------------------------"
  done
done

echo "✅ All jobs completed."
