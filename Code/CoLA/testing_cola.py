from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np


task = "cola"
batch_size = 16

dataset = load_dataset("glue", task)
metric = load_metric("glue", task)

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)


def preprocess(data):
    tokens = tokenizer(data['sentence'], truncation=True)

    return tokens


encoded_data = dataset.map(preprocess, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

args = TrainingArguments(
    f'distilbert-base-uncased---{task}',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='matthews_correlation',
    push_to_hub=False
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_data['train'],
    eval_dataset=encoded_data['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

print(trainer.evaluate())


