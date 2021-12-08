from datasets import load_dataset, load_metric
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          ElectraTokenizerFast, ElectraForSequenceClassification,
                          XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification)
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
import argparse


task = "sst2"
sst = load_dataset("glue", name=task)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='bert', type=str, help="select a pretrained model: "
                                                              "['bert', 'electra', 'xlm-roberta']")
args = parser.parse_args()

if args.model == 'bert':
    checkpoint = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

elif args.model == 'electra':
    checkpoint = 'google/electra-small-discriminator'
    tokenizer = ElectraTokenizerFast.from_pretrained(checkpoint)
    model = ElectraForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

elif args.model == 'xlm-roberta':
    checkpoint = 'roberta-base'
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(checkpoint)
    model = XLMRobertaForSequenceClassification.from_pretrained(checkpoint)


def pretrained_tokenizer(data):
    return tokenizer(data['sentence'], truncation=True)


sst = sst.map(pretrained_tokenizer, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='sst_results',
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

# bert          ->   val_acc = %
# electra       ->   val_acc = 88.18%
# xlm-roberta   ->   val_acc = %
trainer.train()

