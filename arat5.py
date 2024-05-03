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
from transformers import (
    TrainingArguments,
    Trainer
)
import evaluate
from peft import get_peft_model
from arguments import ModelArguments, DataArguments
import wandb
from nltk.tokenize import sent_tokenize
import nltk
from evaluate import load

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
from evaluate import load
from peft import LoraConfig, prepare_model_for_kbit_training

import re
from pathlib import Path
from utils_seq import *
import numpy as np
from peft import PeftModel    
import logging
import os
from huggingface_hub import login
login(token="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")

bleu = evaluate.load("bleu")

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions[0], eval_pred.label_ids
 
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
 
    result = bleu.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )
 
    return {k: v for k, v in result.items()}

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
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="right",)
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



    model = AutoModelForSeq2SeqLM.from_pretrained(
    model_args.model_name_or_path,
    # quantization_config=bnb_config,
    trust_remote_code=True,
    )


   
    save_path = f'{training_args.output_dir}/{model_args.model_name_or_path}'
    training_args.output_dir = save_path

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
    )
     
    history = trainer.train()

    # Save the fine-tuned model
    trainer.save_model(f"{save_path}/best")  # Adjust save directory

    print("Training completed. Model saved. at ", save_path)

    # eval_results = trainer.evaluate(val_dataset)

    # print("Evaluation Results:", eval_results)
    # wandb.log(eval_results)

if __name__ == "__main__":
    main()


