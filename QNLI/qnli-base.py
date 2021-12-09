from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
import torch

torch.cuda.empty_cache()

# Check GPU availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

print(torch.cuda.memory_summary(device=device, abbreviated=False))

# Load data set and metrics
# Task options are
# ["sst2", "mnli", "mnli_mismatched", "mnli_matched", "cola", "stsb", "mrpc", "qqp", "qnli", "rte", "wnli", "hans"]
# Use mnli since it has train dataset. For simplicity, pull test for matched from the same source
task = 'qnli'

metric = load_metric('glue', task)
data = load_dataset('glue', task)

# Look at the data object
print(data)

# Get model for tokenizing and classification
checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint, use_fast=True)

def question_sentence_tokenize(df):
    return tokenizer(df['question'], df['sentence'], truncation=True)

data_tokenized = data.map(question_sentence_tokenize, batched=True)
data_tokenized = data_tokenized.remove_columns(['idx', 'sentence', 'question'])

collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='qnli_classification_results',
    per_device_train_batch_size=16,
    num_train_epochs=2,
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch'
)

sequence_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def compute_model_metrics(eval_preds):
    metric = load_metric("glue", task)
    logits, labels = eval_preds
    logits = logits.flatten()
    return metric.compute(predictions=logits, references=labels)

qnli_trainer = Trainer(
    sequence_model,
    training_args,
    train_dataset=data_tokenized['train'],
    eval_dataset=data_tokenized['validation'],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_model_metrics
)

qnli_trainer.train()
