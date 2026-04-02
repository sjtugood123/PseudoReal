#!/bin/bash
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

MODEL=${MODEL:-"/mnt/model/Meta-Llama-3-8B"}
MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
# Data params
NUM_SEQUENCES=${NUM_SEQUENCES:-128}
# Quantization params
FORMAT=${FORMAT:-"nvfp"}
W_BITS=${W_BITS:-4}
A_BITS=${A_BITS:-4}
W_GROUP_SIZE=${W_GROUP_SIZE:-16}
A_GROUP_SIZE=${A_GROUP_SIZE:-16}
GPTQ=${GPTQ:-0}
W_OBSERVER=${W_OBSERVER:-"minmax"}
QUANTIZATION_ORDER=${QUANTIZATION_ORDER:-"default"}
# Save params
EXPORT_QUANTIZATION=${EXPORT_QUANTIZATION:-""}
# Transform params
TRANSFORM_CLASS=${TRANSFORM_CLASS:-"identity"}
HADAMARD_GROUP_SIZE=${HADAMARD_GROUP_SIZE:-128}
# Evaluation params
EVAL_PERPLEXITY=${EVAL_PERPLEXITY:-1}
EVAL_OPENLLM=${EVAL_OPENLLM:-0}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-"auto"}
# Misc params
LOG_WANDB=${LOG_WANDB:-0}
DTYPE=${DTYPE:-"auto"}
CPU_OFFLOAD_ACTIVATIONS=${CPU_OFFLOAD_ACTIVATIONS:-0}

SCRIPT_ARGS=""

if [[ $GPTQ == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --gptq"
fi

if [[ $EVAL_PERPLEXITY == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_perplexity"
fi

if [[ $EVAL_OPENLLM == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_openllm"
fi

if [[ $LOG_WANDB == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --log_wandb"
fi

METHOD_NAME=""
if [[ $GPTQ == 1 ]]; then
    METHOD_NAME="GPTQ"
else
    METHOD_NAME="RTN"
fi

if [[ $CPU_OFFLOAD_MODULES == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --cpu_offload_modules"
fi

if [[ $CPU_OFFLOAD_ACTIVATIONS == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --cpu_offload_activations"
fi

export WANDB_PROJECT="FP-Quantization-Harness"
export WANDB_NAME=${MODEL}/${FORMAT}-w${W_BITS}-a${A_BITS}-${METHOD_NAME}-${TRANSFORM_CLASS}-transform

if [[ $EXPORT_QUANTIZATION == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --export_quantized_models"
    SAVE_DIR=quantized_models
fi

python model_quant.py \
    --model_name_or_path=${MODEL} \
    --format=${FORMAT} \
    --w_bits=${W_BITS} \
    --a_bits=${A_BITS} \
    --w_group_size=${W_GROUP_SIZE} \
    --a_group_size=${A_GROUP_SIZE} \
    --transform_class=${TRANSFORM_CLASS} \
    --w_observer=${W_OBSERVER} \
    --quantization_order=${QUANTIZATION_ORDER} \
    $SCRIPT_ARGS \
    --hadamard_group_size=${HADAMARD_GROUP_SIZE} \
    --dataset_name_or_path=fineweb-edu \
    --num_sequences=${NUM_SEQUENCES} \
    --sequence_length=2048 \
    --dtype=${DTYPE} \
    --lm_eval_batch_size=${LM_EVAL_BATCH_SIZE} \
    --save_path "/home/xtzhao/PseudoReal/export_fpquant" \
    --export_quantized_model \
    --cpu_offload_activations \
    --cpu_offload_modules \
    --fuse_global_scale \
    --amp
