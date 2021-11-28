from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, AdamW
from transformers import get_scheduler, TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# num of epochs
epochs = 2

# specify glue task and get metric
task = "sst2"
metric = load_metric("glue", task)

# training and validation data
train = load_dataset("glue", name=task, split='train[0:6734]')  # train[0:6734]
validation = load_dataset("glue", name=task, split='validation[0:200]')

# model checkpoint and pretrained tokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

# 6734 - train
# 87 - test


# function to tokenize the sentence in dataset
def preprocess(example):
    return tokenizer(example["sentence"], truncation=True)


# encode the training/validation data
encoded_train = train.map(preprocess, batched=True)
encoded_validation = validation.map(preprocess, batched=True)

# data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# remove sentence and index columns
encoded_train = encoded_train.remove_columns(["sentence", "idx"])
encoded_validation = encoded_validation.remove_columns(["sentence", "idx"])

# change type to a pytorch Tensor
encoded_train.set_format("torch")
encoded_validation.set_format("torch")

# feed training/validation data into DataLoader
train_dataloader = DataLoader(encoded_train, shuffle=True, batch_size=32, collate_fn=data_collator)
val_dataloader = DataLoader(encoded_validation, batch_size=32, collate_fn=data_collator)

# define model (pretrained) and define optimizer
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=0.0005)

# total number of training steps, and learning rate scheduler
num_training_steps = epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# train on GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Device: ', device)

train_steps, val_steps = 0, 0
THRESHOLD = 0.5

# train the model
for epoch in range(epochs):

    model.train()
    pred_logits, real_labels = np.zeros((1, 1)), np.zeros((1, 1))

    with tqdm(total=len(train_dataloader), desc="Training --> ") as pbar:
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            train_loss = outputs.loss
            train_loss.backward()

            train_steps += 1

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {train_loss / train_steps}")

    # evaluate the model
    model.eval()

    with tqdm(total=len(val_dataloader), desc='Validation --> ') as pbar:
        # validate the data
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                val_loss = outputs.loss

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {val_loss / val_steps}")

    print('Validation Accuracy: ', metric.compute(predictions=predictions, references=batch["labels"]))
