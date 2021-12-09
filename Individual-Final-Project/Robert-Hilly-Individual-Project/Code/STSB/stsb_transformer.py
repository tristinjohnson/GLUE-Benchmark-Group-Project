from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import torch
import pandas as pd
import numpy as np
import os

# have our computations run on the GPU
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# specify the Semantic Textual Similarity from GLUE
task = "stsb"

# load in the training, validation, and testing sets
stsb = load_dataset("glue", task)

# get the tokenizer from BERT
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# define a function that will tokenize each pair of questions
def bert_tokenize(dataset):
    return tokenizer(dataset["sentence1"], dataset["sentence2"], truncation=True)

# apply the tokenizer to the training, validation, and testing sets
stsb_tokenized = stsb.map(bert_tokenize, batched=True)

# properly apply padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# specify training arguments
training_args = TrainingArguments(
    output_dir="stsb_seq_classification_results",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch"
)

# bring in the BERT and add in a new head that is suitable for sequence classification
seq_class_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)

# define a function that calculates metrics for our model
def compute_model_metrics(eval_preds):
    metric = load_metric("glue", task)
    logits, labels = eval_preds
    logits = logits.flatten()
    return metric.compute(predictions=logits, references=labels)

# define a blueprint with how our model will be ran
stsb_trainer = Trainer(
    seq_class_model,
    training_args,
    train_dataset=stsb_tokenized["train"],
    eval_dataset=stsb_tokenized["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_model_metrics
)

# validation pearson: 0.873
# validation spearman: 0.869
stsb_trainer.train()

