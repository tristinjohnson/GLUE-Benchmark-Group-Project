from datasets import load_dataset, load_metric
from transformers import (ElectraTokenizerFast, ElectraForSequenceClassification,
                          AlbertTokenizerFast, AlbertForSequenceClassification)
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
import argparse


task = "rte"
sst = load_dataset("glue", name=task)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='electra', type=str, help="select a pretrained model: ['electra', 'albert']")
args = parser.parse_args()

if args.model == 'electra':
    checkpoint = 'google/electra-small-discriminator'
    tokenizer = ElectraTokenizerFast.from_pretrained(checkpoint)
    model = ElectraForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

elif args.model == 'albert':
    checkpoint = 'albert-base-v2'
    tokenizer = AlbertTokenizerFast.from_pretrained(checkpoint)
    model = AlbertForSequenceClassification.from_pretrained(checkpoint)


def pretrained_tokenizer(data):
    return tokenizer(data['sentence1'], data['sentence2'], truncation=True)


sst = sst.map(pretrained_tokenizer, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='rte_results',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    evaluation_strategy='epoch'
)


def compute_metrics(pred):
    metric = load_metric("glue", task)
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    training_args,
    train_dataset=sst['train'],
    eval_dataset=sst['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# electra ->   val_acc = 63.176%, :58 per epoch
# albert   ->  val_acc = 54.151%, 2:40 per epoch
trainer.train()

