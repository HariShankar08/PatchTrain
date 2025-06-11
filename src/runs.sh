#!/bin/bash

echo "Running Patch Training with K=4, Llama 3.2 1B Instruct on dataset: WMT-FR"
python main.py --mode patch_train --model_path meta-llama/Llama-3.2-1B-Instruct --dataset wmt14_fr-en --save_path output/llama32_1b_k4_2p1f | tee logs/llama32_1b_k4_2p1f.log

echo "Running Full Training with Llama 3.2 1B Instruct on dataset: WMT-FR"
python main.py --mode full_finetune --model_path meta-llama/Llama-3.2-1B-Instruct --dataset wmt14_fr-en --save_path output/llama32_1b_fft | tee logs/llama32_1b_fft.log

echo "Running Patch Training with K=2, Llama 3.2 1B Instruct on dataset: WMT-FR"
python main.py --mode patch_train --patch_size 2 --model_path meta-llama/Llama-3.2-1B-Instruct --dataset wmt14_fr-en --save_path output/llama32_1b_k2_2p1f | tee logs/llama32_1b_k2_2p1f.log

echo "Running Patch Training with K=8, Llama 3.2 1B Instruct on dataset: WMT-FR"
python main.py --mode patch_train --patch_size 8 --model_path meta-llama/Llama-3.2-1B-Instruct --dataset wmt14_fr-en --save_path output/llama32_1b_k2_2p1f | tee logs/llama32_1b_k2_2p1f.log

echo "Running Patch Training with K=16, Llama 3.2 1B Instruct on dataset: WMT-FR"
python main.py --mode patch_train --patch_size 16 --model_path meta-llama/Llama-3.2-1B-Instruct --dataset wmt14_fr-en --save_path output/llama32_1b_k2_2p1f | tee logs/llama32_1b_k2_2p1f.log






