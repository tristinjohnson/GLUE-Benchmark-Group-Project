from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import torch

# Check GPU availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set batch size to use for our model
batch_size = 32

# Load data set and metrics
# Task options are
# ["sst2", "mnli", "mnli_mismatched", "mnli_matched", "cola", "stsb", "mrpc", "qqp", "qnli", "rte", "wnli", "hans"]
# Use mnli since it has train dataset. For simplicity, pull test for matched from the same source
task = 'mnli'

metric = load_metric('glue', task)
# Get train and test as datatype <class 'datasets.arrow_dataset.Dataset'>
train = load_dataset('glue', task, split='train[0:30000]') # Full sample is 392702 but only taking 30000
test = load_dataset('glue', task, split='test_matched') # Full sample is 9796, getting all of the sample

# Load model
checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint, use_fast=True)

# Tokenize train and test
# Append the two together with specific tokens
premise = train['premise']
hypothesis = train['hypothesis']
label = train['label']

token_list = []
seq_list = []
mask_list = []
y = []

# Loop through premise and hypothesis, create tokens based on the concatenation of both
# add_special_tokens=True adds cls_token id to start (101) and sep_token_id (102) to end of each seq 1 and seq 2
for p, h in zip(premise, hypothesis):
    tok_info = tokenizer(p, h, add_special_tokens=True)
    token_id, seq_id, mask_id = tok_info['input_ids'], tok_info['token_type_ids'], tok_info['attention_mask']
    # Append token, seq info, and mask info to list as tensors
    token_list.append(torch.tensor(token_id))
    seq_list.append(torch.tensor(seq_id))
    mask_list.append(torch.tensor(mask_id))

# Turn label into target variable for our model
# Get label coded using a dictionary
label_dict = {'entailment': 0,
              'contradiction': 1,
              'neutral': 2}

for val in label:
    y.append(label_dict[val])

y = torch.tensor(y)

# Load data into batches


# Create model

# Model needs tokens, sequence_ids, mask_ids, and label (turned to a factor)