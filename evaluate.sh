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

python eval.py \
--model_name_or_path='core42/jais-13b' \
--wandb_project="nadi_sharedtask" \
--wandb_run_name="jais" \
--dataset boda/nadi2024 \
--use_flash_attention_2=False \
--prompt_key="prompt" \
--chunk_size=100 \
--split="dev" \
--per_device_eval_batch_size=4 \
--save_file='outputs/jais_val' \
--checkpoint_path='/l/users/abdelrahman.sadallah/nadi/core42/jais-13b/best/' \
--output_dir=$SAVEDIR 
echo "ending"
