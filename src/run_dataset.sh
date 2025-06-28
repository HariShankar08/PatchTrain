#!/bin/bash

MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT='llama32_1b'
DATASET_NAME="wmt_fr"  # WMT-FR dataset
BATCH_SIZE=8
NUM_RUNS=1

PATCH_SIZE=4
LAMBDA_RATIO=0.667

run_type=0

if [ $run_type == 0 ]; then
# Run Patch Training
echo "Running Patch Training with K=$PATCH_SIZE, lambda=$LAMBDA_RATIO ($MODEL_SHORT on $DATASET_NAME) \
    with Batch Size=$BATCH_SIZE, \
    Num Runs=$NUM_RUNS"

python main.py --mode patch_train \
    --patch_size $PATCH_SIZE \
    --lambda_ratio $LAMBDA_RATIO \
    --model_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --batch_size $BATCH_SIZE \
    --num_runs $NUM_RUNS | tee logs/${MODEL_SHORT}_k${PATCH_SIZE}_lambda${LAMBDA_RATIO//./}.log
fi

if [ $run_type == 1 ]; then

# Run Full Finetune Baseline
echo "Running Full Finetune Baseline ($MODEL_PATH) on $DATASET_NAME \
    with Batch Size=$BATCH_SIZE, \
    Num Runs=$NUM_RUNS"

python main.py --mode full_finetune \
    --model_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --batch_size $BATCH_SIZE \
    --num_runs $NUM_RUNS | tee logs/${MODEL_SHORT}_full_finetune.log

fi

