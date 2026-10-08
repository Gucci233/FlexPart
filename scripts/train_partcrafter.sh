#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")/.."

NUM_MACHINES=${NUM_MACHINES:-1}
NUM_LOCAL_GPUS=${NUM_LOCAL_GPUS:-8}
MACHINE_RANK=${MACHINE_RANK:-0}

export TORCH_CPP_LOG_LEVEL=ERROR

accelerate launch \
    --num_machines $NUM_MACHINES \
    --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
    --machine_rank $MACHINE_RANK \
    src/train_partcrafter.py \
        --config configs/mp8_nt512.yaml \
        --use_ema \
        --gradient_accumulation_steps 4 \
        --output_dir ./ \
        --tag my_model \
        --pin_memory \
        --allow_tf32 \
        "$@"
