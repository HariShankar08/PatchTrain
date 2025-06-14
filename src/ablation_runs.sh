#!/bin/bash

echo "Running Patch Training with K=8, lambda=0.5 (Llama 3.2 1B Instruct on WMT-FR)"
python main.py --mode patch_train \
    --patch_size 8 \
    --lambda_ratio 0.5 \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset wmt_fr \
    --save_path output/llama32_1b_k8_lambda05 \
    --batch_size 8 \
    --num_runs 5 | tee logs/llama32_1b_k8_lambda05.log

echo "Running Patch Training with K=8, lambda=0.75 (Llama 3.2 1B Instruct on WMT-FR)"
python main.py --mode patch_train \
    --patch_size 8 \
    --lambda_ratio 0.75 \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset wmt_fr \
    --save_path output/llama32_1b_k8_lambda075 \
    --batch_size 8 \
    --num_runs 5 | tee logs/llama32_1b_k8_lambda075.log

echo "Running Patch Training with K=8, lambda=0.8 (Llama 3.2 1B Instruct on WMT-FR)"
python main.py --mode patch_train \
    --patch_size 8 \
    --lambda_ratio 0.8 \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset wmt_fr \
    --save_path output/llama32_1b_k8_lambda08 \
    --batch_size 8 \
    --num_runs 5 | tee logs/llama32_1b_k8_lambda08.log 

echo "Running Patch Training with K=8, lambda=1.0 (Llama 3.2 1B Instruct on WMT-FR)"
python main.py --mode patch_train \
    --patch_size 8 \
    --lambda_ratio 1.0 \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --dataset wmt_fr \
    --save_path output/llama32_1b_k8_lambda08 \
    --batch_size 8 \
    --num_runs 5 | tee logs/llama32_1b_k8_lambda08.log 
