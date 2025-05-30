#!/bin/bash

echo "Running Patch Training on OPT-125M"
python main.py --mode patch_train --model_path facebook/opt-125m --num_runs 5 --save_path output/opt125_k4_2p1f | tee logs/opt125_k4_2p1f.log

echo "Running Full Training on OPT-125M"
python main.py --mode full_finetune --model_path facebook/opt-125m --num_runs 5 --save_path output/opt125_fft | tee logs/opt125_fft.log

echo "Running Patch Training on OPT-350M"
python main.py --mode patch_train --model_path facebook/opt-350m --num_runs 5 --save_path output/opt350_k4_2p1f | tee logs/opt350_k4_2p1f.log

echo "Running Full Training on OPT-350M"
python main.py --mode full_finetune --model_path facebook/opt-350m --num_runs 5 --save_path output/opt350_fft | tee logs/opt350_fft.log