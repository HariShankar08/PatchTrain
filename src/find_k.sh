#!/bin/bash

MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct". # Path to the model
MODEL_SHORT='llama32_1b'. # Short name for the model - only for your convenience
DATASET_NAME="wmt_fr"  # WMT-FR dataset <- Replace with "wmt_hi" or "cnn"
BATCH_SIZE=8. # Batch size for training - in case it breaks for larger models, you can reduce it to 4 or 2
NUM_RUNS=5

for PATCH_SIZE in (2 4 8 16); do
    echo "Running Patch Training with K=$PATCH_SIZE ($MODEL_SHORT on $DATASET_NAME) \
        with Batch Size=$BATCH_SIZE, \
        Num Runs=$NUM_RUNS"
    python main.py --mode patch_train \
        --patch_size $PATCH_SIZE \
        --model_path $MODEL_PATH \
        --dataset $DATASET_NAME \
        --batch_size $BATCH_SIZE \
        --num_runs $NUM_RUNS | tee logs/${MODEL_SHORT}_k${PATCH_SIZE}.log
    echo "Completed Patch Training with K=$PATCH_SIZE"
    echo "----------------------------------------"
done

