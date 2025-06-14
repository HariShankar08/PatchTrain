#!/bin/bash

# # WMT_FR here, WMT_HI, CNN commented out. (No changes except for DATASET_NAME)

# Globals
PATCH_SIZE=8
BATCH_SIZE=8
LAMBDA_RATIO=0.8  # May have to change this, based on ablation_runs result. 
NUM_RUNS=5
DATASET_NAME="wmt_fr"  

MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT='llama32_1b'

# Llama 3.2 1B Instruct

echo "Running LoRA Baseline ($MODEL_PATH) 
on $DATASET_NAME \
with Batch Size=$BATCH_SIZE, \ 
Num Runs=$NUM_RUNS"

python main.py \
    --mode lora_finetune \
    --model_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --save_path $MODEL_SHORT'_bs'$BATCH_SIZE \
    --batch_size $BATCH_SIZE \

echo "Running Patch Training ($MODEL_PATH) on $DATASET_NAME with K=$PATCH_SIZE, \
Lambda Ratio=$LAMBDA_RATIO, \
Batch Size=$BATCH_SIZE, \
Num Runs=$NUM_RUNS"
python main.py \
    --mode patch_peft \
    --model_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --save_path $MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --num_runs $NUM_RUNS \  
    | tee logs/$MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE.log

# Phi 3.5 Mini Instruct

MODEL_PATH="microsoft/Phi-3.5-mini-instruct"
MODEL_SHORT='phi35_mini'


echo "Running LoRA Baseline ($MODEL_PATH) 
on $DATASET_NAME \
with Batch Size=$BATCH_SIZE, \ 
Num Runs=$NUM_RUNS"

python main.py \
    --mode lora_finetune \
    --model_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --save_path $MODEL_SHORT'_bs'$BATCH_SIZE \
    --batch_size $BATCH_SIZE \

echo "Running Patch Training ($MODEL_PATH) on $DATASET_NAME with K=$PATCH_SIZE, \
Lambda Ratio=$LAMBDA_RATIO, \
Batch Size=$BATCH_SIZE, \
Num Runs=$NUM_RUNS"
python main.py \
    --mode patch_peft \
    --model_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --save_path $MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --num_runs $NUM_RUNS \  
    | tee logs/$MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE.log


# Falcon H1 7B

MODEL_PATH="tiiuae/Falcon-H1-7B-Instruct"
MODEL_SHORT='falcon7b'


echo "Running LoRA Baseline ($MODEL_PATH) 
on $DATASET_NAME \
with Batch Size=$BATCH_SIZE, \ 
Num Runs=$NUM_RUNS"

python main.py \
    --mode lora_finetune \
    --model_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --save_path $MODEL_SHORT'_bs'$BATCH_SIZE \
    --batch_size $BATCH_SIZE \

echo "Running Patch Training ($MODEL_PATH) on $DATASET_NAME with K=$PATCH_SIZE, \
Lambda Ratio=$LAMBDA_RATIO, \
Batch Size=$BATCH_SIZE, \
Num Runs=$NUM_RUNS"
python main.py \
    --mode patch_peft \
    --model_path $MODEL_PATH \
    --dataset $DATASET_NAME \
    --save_path $MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --num_runs $NUM_RUNS \  
    | tee logs/$MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE.log



# #!/bin/bash

# # Globals
# PATCH_SIZE=4
# BATCH_SIZE=8
# LAMBDA_RATIO=0.6666
# NUM_RUNS=5
# DATASET_NAME="wmt_hi"  

# MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_SHORT='llama32_1b'

# # Llama 3.2 1B Instruct

# echo "Running LoRA Baseline ($MODEL_PATH) 
# on $DATASET_NAME \
# with Batch Size=$BATCH_SIZE, \ 
# Num Runs=$NUM_RUNS"

# python main.py \
#     --mode lora_finetune \
#     --model_path $MODEL_PATH \
#     --dataset $DATASET_NAME \
#     --save_path $MODEL_SHORT'_bs'$BATCH_SIZE \
#     --batch_size $BATCH_SIZE \

# echo "Running Patch Training ($MODEL_PATH) on $DATASET_NAME with K=$PATCH_SIZE, \
# Lambda Ratio=$LAMBDA_RATIO, \
# Batch Size=$BATCH_SIZE, \
# Num Runs=$NUM_RUNS"
# python main.py \
#     --mode patch_peft \
#     --model_path $MODEL_PATH \
#     --dataset $DATASET_NAME \
#     --save_path $MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE \
#     --batch_size $BATCH_SIZE \
#     --num_runs $NUM_RUNS \  
#     | tee logs/$MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE.log

# # Phi 3.5 Mini Instruct

# MODEL_PATH="microsoft/Phi-3.5-mini-instruct"
# MODEL_SHORT='phi35_mini'


# echo "Running LoRA Baseline ($MODEL_PATH) 
# on $DATASET_NAME \
# with Batch Size=$BATCH_SIZE, \ 
# Num Runs=$NUM_RUNS"

# python main.py \
#     --mode lora_finetune \
#     --model_path $MODEL_PATH \
#     --dataset $DATASET_NAME \
#     --save_path $MODEL_SHORT'_bs'$BATCH_SIZE \
#     --batch_size $BATCH_SIZE \

# echo "Running Patch Training ($MODEL_PATH) on $DATASET_NAME with K=$PATCH_SIZE, \
# Lambda Ratio=$LAMBDA_RATIO, \
# Batch Size=$BATCH_SIZE, \
# Num Runs=$NUM_RUNS"
# python main.py \
#     --mode patch_peft \
#     --model_path $MODEL_PATH \
#     --dataset $DATASET_NAME \
#     --save_path $MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE \
#     --batch_size $BATCH_SIZE \
#     --num_runs $NUM_RUNS \  
#     | tee logs/$MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE.log


# # Falcon H1 7B

# MODEL_PATH="tiiuae/Falcon-H1-7B-Instruct"
# MODEL_SHORT='falcon7b'


# echo "Running LoRA Baseline ($MODEL_PATH) 
# on $DATASET_NAME \
# with Batch Size=$BATCH_SIZE, \ 
# Num Runs=$NUM_RUNS"

# python main.py \
#     --mode lora_finetune \
#     --model_path $MODEL_PATH \
#     --dataset $DATASET_NAME \
#     --save_path $MODEL_SHORT'_bs'$BATCH_SIZE \
#     --batch_size $BATCH_SIZE \

# echo "Running Patch Training ($MODEL_PATH) on $DATASET_NAME with K=$PATCH_SIZE, \
# Lambda Ratio=$LAMBDA_RATIO, \
# Batch Size=$BATCH_SIZE, \
# Num Runs=$NUM_RUNS"
# python main.py \
#     --mode patch_peft \
#     --model_path $MODEL_PATH \
#     --dataset $DATASET_NAME \
#     --save_path $MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE \
#     --batch_size $BATCH_SIZE \
#     --num_runs $NUM_RUNS \  
#     | tee logs/$MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE.log






# #!/bin/bash

# # Globals
# PATCH_SIZE=4
# BATCH_SIZE=8
# LAMBDA_RATIO=8
# NUM_RUNS=5
# DATASET_NAME="wmt_cnn"  

# MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_SHORT='llama32_1b'

# # Llama 3.2 1B Instruct

# echo "Running LoRA Baseline ($MODEL_PATH) 
# on $DATASET_NAME \
# with Batch Size=$BATCH_SIZE, \ 
# Num Runs=$NUM_RUNS"

# python main.py \
#     --mode lora_finetune \
#     --model_path $MODEL_PATH \
#     --dataset $DATASET_NAME \
#     --save_path $MODEL_SHORT'_bs'$BATCH_SIZE \
#     --batch_size $BATCH_SIZE \

# echo "Running Patch Training ($MODEL_PATH) on $DATASET_NAME with K=$PATCH_SIZE, \
# Lambda Ratio=$LAMBDA_RATIO, \
# Batch Size=$BATCH_SIZE, \
# Num Runs=$NUM_RUNS"
# python main.py \
#     --mode patch_peft \
#     --model_path $MODEL_PATH \
#     --dataset $DATASET_NAME \
#     --save_path $MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE \
#     --batch_size $BATCH_SIZE \
#     --num_runs $NUM_RUNS \  
#     | tee logs/$MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE.log

# # Phi 3.5 Mini Instruct

# MODEL_PATH="microsoft/Phi-3.5-mini-instruct"
# MODEL_SHORT='phi35_mini'


# echo "Running LoRA Baseline ($MODEL_PATH) 
# on $DATASET_NAME \
# with Batch Size=$BATCH_SIZE, \ 
# Num Runs=$NUM_RUNS"

# python main.py \
#     --mode lora_finetune \
#     --model_path $MODEL_PATH \
#     --dataset $DATASET_NAME \
#     --save_path $MODEL_SHORT'_bs'$BATCH_SIZE \
#     --batch_size $BATCH_SIZE \

# echo "Running Patch Training ($MODEL_PATH) on $DATASET_NAME with K=$PATCH_SIZE, \
# Lambda Ratio=$LAMBDA_RATIO, \
# Batch Size=$BATCH_SIZE, \
# Num Runs=$NUM_RUNS"
# python main.py \
#     --mode patch_peft \
#     --model_path $MODEL_PATH \
#     --dataset $DATASET_NAME \
#     --save_path $MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE \
#     --batch_size $BATCH_SIZE \
#     --num_runs $NUM_RUNS \  
#     | tee logs/$MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE.log


# # Falcon H1 7B

# MODEL_PATH="tiiuae/Falcon-H1-7B-Instruct"
# MODEL_SHORT='falcon7b'


# echo "Running LoRA Baseline ($MODEL_PATH) 
# on $DATASET_NAME \
# with Batch Size=$BATCH_SIZE, \ 
# Num Runs=$NUM_RUNS"

# python main.py \
#     --mode lora_finetune \
#     --model_path $MODEL_PATH \
#     --dataset $DATASET_NAME \
#     --save_path $MODEL_SHORT'_bs'$BATCH_SIZE \
#     --batch_size $BATCH_SIZE \

# echo "Running Patch Training ($MODEL_PATH) on $DATASET_NAME with K=$PATCH_SIZE, \
# Lambda Ratio=$LAMBDA_RATIO, \
# Batch Size=$BATCH_SIZE, \
# Num Runs=$NUM_RUNS"
# python main.py \
#     --mode patch_peft \
#     --model_path $MODEL_PATH \
#     --dataset $DATASET_NAME \
#     --save_path $MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE \
#     --batch_size $BATCH_SIZE \
#     --num_runs $NUM_RUNS \  
#     | tee logs/$MODEL_SHORT'_k'$PATCH_SIZE'_l'$LAMBDA_RATIO'_bs'$BATCH_SIZE.log











