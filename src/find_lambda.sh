#!/bin/bash

MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT='llama32_1b'
DATASET_NAME="wmt_fr"  # WMT-FR dataset
BATCH_SIZE=8
NUM_RUNS=5

PATCH_SIZE=8

for ratio in (0.5 0.75 0.8 1.0); do
    echo "Running Patch Training with K=8, lambda=$ratio (Llama 3.2 1B Instruct on WMT-FR)"
    python main.py --mode patch_train \
        --patch_size $PATCH_SIZE \
        --lambda_ratio $ratio \
        --model_path $MODEL_PATH \
        --dataset $DATASET_NAME \
        --batch_size $BATCH_SIZE \
        --num_runs $NUM_RUNS | tee logs/${MODEL_SHORT}_k${PATCH_SIZE}_lambda${ratio//./}.log
    echo "Completed Patch Training with K=8, lambda=$ratio"
    echo "----------------------------------------"
done

