import re
from datasets import load_dataset
import pyarabic.araby as araby
from transformers import AutoTokenizer


# DEFAULT_ARABIC_SYSTEM_PROMPT = '''
# The following is a sentence in {dialect} Arabic dialect. Please translate it to Modern Standard Arabic (MSA).
# '''.strip()


def clean_text(text):
    '''
    Cleans text from unnecessary characters.
    '''
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)

    return re.sub(r'\^[^ ]+', '', text)


def print_trainable_parameters(model):
    '''
    Prints the number of trainable parameters in the model.
    '''
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}'
    )



def generate_arabic_training_prompt(example, tokenizer, field='prompt'):

    source = example['source']
    target = example['target']
    dialect = example['dialect']

    DEFAULT_ARABIC_SYSTEM_PROMPT = '''
    The following is a sentence in {dialect} Arabic dialect. Please translate it to Modern Standard Arabic (MSA).
    '''.strip()

    DEFAULT_ARABIC_SYSTEM_PROMPT = DEFAULT_ARABIC_SYSTEM_PROMPT.format(dialect=dialect)

    prompt = f'''
### Instruction: {DEFAULT_ARABIC_SYSTEM_PROMPT}

### Input:
{source}

### Response:
'''.strip()
    
    # if train:

    #     prompt = prompt + f'{target}'
    MAX_LENGTH = 512
    example[field] =  prompt
    model_inputs = tokenizer(
        example[field],
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length'
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length'
        )
 
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_dataset(dataset_name='boda/nadi2024',split = 'train', field='prompt'):
    '''
    Returns train, validation and test datasets for arabic, in the format described in
    generate_arabic_training_prompt().
    '''
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/AraT5v2-base-1024")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name,split=split)

    dataset = dataset.map(generate_arabic_training_prompt, fn_kwargs={'field': field, 'tokenizer':tokenizer})
    
    return dataset
