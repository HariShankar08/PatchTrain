#!/bin/bash


echo "Running Patch Training with K=4, Llama 3.2 1B Instruct on dataset: WMT-FR with batch size 4"
python main.py --mode patch_peft --model_path meta-llama/Llama-3.2-1B-Instruct --dataset wmt_hi --save_path output/llama32_1b_k4_2p1f --batch_size 8 --num_runs 5 | tee logs/llama32_1b_k4_2p1f_bs4.log

echo "Running Full Training with Llama 3.2 1B Instruct on dataset: WMT-FR with batch size 4"
python main.py --mode lora_finetune --model_path meta-llama/Llama-3.2-1B-Instruct --dataset wmt_hi --save_path output/llama32_1b_fft --batch_size 8 --num_runs 5 | tee logs/llama32_1b_fft_bs4.log

echo "Running Patch Training with K=2, Llama 3.2 1B Instruct on dataset: WMT-FR with batch size 4"
python main.py --mode patch_peft --patch_size 2 --model_path meta-llama/Llama-3.2-1B-Instruct --dataset wmt_hi --save_path output/llama32_1b_k2_2p1f --batch_size 8 --num_runs 5 | tee logs/llama32_1b_k2_2p1f_bs4.log

echo "Running Patch Training with K=8, Llama 3.2 1B Instruct on dataset: WMT-FR with batch size 4"
python main.py --mode patch_peft --patch_size 8 --model_path meta-llama/Llama-3.2-1B-Instruct --dataset wmt_hi --save_path output/llama32_1b_k8_2p1f --batch_size 8 --num_runs 5 | tee logs/llama32_1b_k8_2p1f_bs4.log

echo "Running Patch Training with K=16, Llama 3.2 1B Instruct on dataset: WMT-FR with batch size 4"
python main.py --mode patch_peft --patch_size 16 --model_path meta-llama/Llama-3.2-1B-Instruct --dataset wmt_hi --save_path output/llama32_1b_k16_2p1f --batch_size 8 --num_runs 5 | tee logs/llama32_1b_k16_2p1f_bs4.log




