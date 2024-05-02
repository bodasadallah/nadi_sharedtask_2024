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


from utils import *
import numpy as np
from peft import PeftModel    
import logging
import os

# import evaluate 
from evaluate import load 
from torch.utils.data import DataLoader
from tqdm import tqdm


def split_sequence(sequence, chunk_size):
    chunks=[]
    for i in range(0, len(sequence), chunk_size):
        chunks.append(sequence[i: i + chunk_size])
    return chunks
		

def calc_results(prediction, truth, save_file, chunk_size=100):
    

    global bleu_score
    
    if (len(truth) != len(prediction)):
        print ("both files must have same number of instances")
        exit()

    truth_chunks= split_sequence(truth, chunk_size)

    truth_Egyptain=truth_chunks[0]
    truth_Emirati=truth_chunks[1]
    truth_Jordanian=truth_chunks[2]
    truth_Palestinian=truth_chunks[3]

    prediction_chunks= split_sequence(prediction, chunk_size)

    prediction_Egyptain=prediction_chunks[0]
    prediction_Emirati=prediction_chunks[1]
    prediction_Jordanian=prediction_chunks[2]
    prediction_Palestinian=prediction_chunks[3]

    ### get scores
    results_Egyptain = bleu_score.compute(predictions=prediction_Egyptain, references=truth_Egyptain)
    results_Emirati = bleu_score.compute(predictions=prediction_Emirati, references=truth_Emirati)
    results_Jordanian = bleu_score.compute(predictions=prediction_Jordanian, references=truth_Jordanian)
    results_Palestinian = bleu_score.compute(predictions=prediction_Palestinian, references=truth_Palestinian)
    overall_results = bleu_score.compute(predictions=prediction, references=truth)

    #write to a text file
    print('Scores:')
    scores = {
            'Overall': overall_results['bleu']*100,
            'Egyptain': results_Egyptain['bleu']*100,
            'Emirati': results_Emirati['bleu']*100,
            'Jordanian': results_Jordanian['bleu']*100,
            'Palestinian': results_Palestinian['bleu']*100, 
            }
    print(scores)

    with open(save_file, 'w') as score_file:
        score_file.write("Overall: %0.12f\n" % scores["Overall"])
        score_file.write("Egyptain: %0.12f\n" % scores["Egyptain"])
        score_file.write("Emirati: %0.12f\n" % scores["Emirati"])
        score_file.write("Jordanian: %0.12f\n" % scores["Jordanian"])
        score_file.write("Palestinian: %0.12f\n" % scores["Palestinian"])


def inference(prompts, tokenizer, model):
    
   
    encoding = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **encoding,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
        )  

    answer_tokens = outputs[:, encoding.input_ids.shape[1] :]
    output_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

    

    return output_text
        

if __name__ == "__main__":




    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    bleu_score = load("bleu")


    print(f"Loading the   {data_args.split} datasets")
    dataset = get_dataset(
        dataset_name = data_args.dataset,
        split=data_args.split,
        field=data_args.prompt_key)


    save_file = model_args.save_file


    


    val_dataloader = DataLoader(dataset, batch_size=Seq2SeqTrainingArguments.per_device_eval_batch_size, shuffle=False)  


    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        return_dict=True,
        load_in_8bit=True,
        device_map="auto",
    )

    if model_args.checkpoint_path:
        print(f'Loading model from {model_args.checkpoint_path}')
        adapter_checkpoint  = model_args.checkpoint_path
        model = PeftModel.from_pretrained(model, adapter_checkpoint)

    else:
        print(f'Loading Base Model {model_args.model_name_or_path}')





    model = model.eval()


    # Define PAD Token = BOS Token
    model.config.pad_token_id = model.config.bos_token_id


    predictions = []
    labels = []


    torch.cuda.empty_cache()

 
    for batch in tqdm(val_dataloader):

        prompts = batch['prompt']
        ans = []

        labels.extend(batch['target'])

        output_text, answer_lengths = inference(prompts=prompts, tokenizer=tokenizer, model=model)

        predictions.extend(output_text)


    assert (len(predictions) == len(labels))



    save_file =   data_args.save_file + '_results.txt'

    preds_file = data_args.save_file + '_predictions.txt'

    with open(preds_file, 'w') as f:
        for item in predictions:
            f.write("%s\n" % item)

    calc_results(predictions, labels, save_file,data_args.chunk_size)
        
