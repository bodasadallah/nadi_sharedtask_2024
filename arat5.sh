#!/bin/bash

#SBATCH --job-name=nlp702 # Job name
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=256GB                  # Total RAM to be used
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH --qos=gpu-8 
#SBATCH -p gpu                      # Use the gpu partition
##SBATCH --nodelist=ws-l6-017


SAVEDIR="/l/users/$USER/nadi"
export HF_CACHE_DIR="/l/users/$USER/hugging_face"

# --num_train_epochs=10 \
export HF_HOME="/l/users/$USER" 

MODEL="core42/jais-13b-chat"

python arat5.py \
--output_dir="$SAVEDIR" \
--model_name_or_path="UBC-NLP/AraT5v2-base-1024" \
--wandb_project="nadi_sharedtask" \
--wandb_run_name="AraT5v2" \
--dataset boda/nadi2024 \
--evaluation_strategy steps \
--max_steps 5000 \
--save_steps=100 \
--eval_steps=100 \
--logging_steps=100 \
--report_to="all" \
--load_best_model_at_end True \
--save_total_limit 3 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--metric_for_best_model loss \
--warmup_ratio=0.07 \
--weight_decay=0.01 \
--learning_rate=1e-06 \
--use_flash_attention_2=False \
--prompt_key="prompt" 
