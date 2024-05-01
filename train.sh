#!/bin/bash

#SBATCH --job-name=NLP804 # Job name
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

python finetune.py \
--output_dir="$SAVEDIR" \
--model_name_or_path='core42/jais-13b' \
--wandb_project="nadi_sharedtask" \
--wandb_run_name="jais" \
--dataset boda/nadi2024 \
--evaluation_strategy steps \
--max_steps 100000 \
--save_steps=500 \
--eval_steps=500 \
--logging_steps=100 \
--report_to="all" \
--load_best_model_at_end True \
--save_total_limit 3 \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=4 \
--metric_for_best_model loss \
--warmup_ratio=0.07 \
--weight_decay=0.01 \
--learning_rate=1e-06 \
--use_flash_attention_2=False \
--prompt_key="prompt" 

echo "ending "
