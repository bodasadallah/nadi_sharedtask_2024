import torch

from transformers import HfArgumentParser, Seq2SeqTrainingArguments,EarlyStoppingCallback

import logging

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from datasets import load_dataset, concatenate_datasets,Value
import numpy as np
from typing import Union, Optional
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset, AutoModel
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    #glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from arguments import ModelArguments, DataArguments
import wandb
from nltk.tokenize import sent_tokenize
import nltk


nltk.download("punkt")
logger = logging.getLogger(__name__)
from transformers import (RobertaForMultipleChoice, RobertaTokenizer, Trainer,
                          TrainingArguments, XLMRobertaForMultipleChoice,
                          XLMRobertaTokenizer)

import pathlib
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from transformers import TrainingArguments
from trl import SFTTrainer

from peft import LoraConfig, prepare_model_for_kbit_training

import re
from pathlib import Path
from utils import *
import numpy as np
from peft import PeftModel    
import logging
import os



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    for arg in vars(model_args):
        print(arg, getattr(model_args, arg))
    for arg in vars(data_args):
        print(arg, getattr(data_args, arg))
    for arg in vars(training_args):
        print(arg, getattr(training_args, arg))


    wandb.init(project=model_args.wandb_project,name=model_args.wandb_run_name)

    ## load tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'right'
    # tokenizer.padding_side  = 'left'
    # model = AutoModel.from_pretrained(model_args.model_name_or_path)



    print("Loading the datasets")
    train_dataset = get_dataset(
        dataset_name = data_args.dataset,
        split='train',
        field=data_args.prompt_key)
        
    val_dataset = get_dataset(
        dataset_name = data_args.dataset,
        split='dev',
        field=data_args.prompt_key,)


    ## Bits and Bytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_flash_attention_2=model_args.use_flash_attention_2,
    )






   
    save_path = f'{training_args.output_dir}/{model_args.model_name_or_path}'
    training_args.output_dir = save_path


    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64
    lora_target_modules = [
                                'c_attn', 'c_proj', 'c_fc', 'c_fc2'
                            ]
    max_seq_length = 256

    peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules = lora_target_modules
        )


    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field=data_args.prompt_key,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        # compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)])
    

    # Start training
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(f"{save_path}/best")  # Adjust save directory

    print("Training completed. Model saved. at ", save_path)

    # eval_results = trainer.evaluate(val_dataset)

    # print("Evaluation Results:", eval_results)
    # wandb.log(eval_results)

if __name__ == "__main__":
    main()


