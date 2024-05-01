import re
from datasets import load_dataset
import pyarabic.araby as araby



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



def generate_arabic_training_prompt(example, field='prompt', train=True):

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
    
    if train:

        prompt = prompt + f'{target}'

    example[field] =  prompt

    return example


def get_dataset(dataset_name='boda/nadi2024',split = 'train', field='prompt'):
    '''
    Returns train, validation and test datasets for arabic, in the format described in
    generate_arabic_training_prompt().
    '''


    dataset = load_dataset(dataset_name,split=split)

    dataset = dataset.map(generate_arabic_training_prompt, fn_kwargs={'field': field, 'train': split == 'train'})
    
    return dataset
