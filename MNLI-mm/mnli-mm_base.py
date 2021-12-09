from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AdamW
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Hyper parameters
n_epochs = 2
learning_rate = 0.0005
batch_size=8

# Check GPU availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# Load data sets
# Task options are
# ["sst2", "mnli", "mnli_mismatched", "mnli_matched", "cola", "stsb", "mrpc", "qqp", "qnli", "rte", "wnli", "hans"]
# Use mnli for training.
# Test on MNLI matched, MNLI mismatched, and ax (diagnostics) as they all have the same format
task1 = 'mnli'
task2 = 'mnli_matched'
task3 = 'mnli_mismatched'
task4 = 'ax'

mnli = load_dataset('glue', task1, split='train[0:8000]')
matched = load_dataset('glue', task2, split='validation')
mismatched = load_dataset('glue', task3, split='validation')
diagnostics = load_dataset('glue', task4, split='test')

print(mnli)
print(matched)
print(mismatched)
print(diagnostics)

# Load BERT model for tokenizing
checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint, use_fast=True)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Tokenize function for hypothesis and premise
def tokenize_ph(data):
    return tokenizer(data['premise'], data['hypothesis'], truncation=True, padding=True)

# Since we are processing all the datasets the same way, write one function to apply to each
def prep_data(data):

    encoded = data.map(tokenize_ph, batched=True)

    # Remove raw data columns as data is encoded in separate columns
    # Remove idx as it does not provide value to this task
    encoded = encoded.remove_columns(['hypothesis', 'premise', 'idx'])

    # Set to torch format
    encoded.set_format("torch")

    dataloader_object = DataLoader(encoded, batch_size=batch_size, collate_fn=collator)

    return dataloader_object

# Get dataloaders for model
mnli_dataloader = prep_data(mnli)
matched_dataloader = prep_data(matched)
mismatched_dataloader = prep_data(mismatched)
diagnostics_dataloader = prep_data(diagnostics)


# Prep model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
optimizer = AdamW(model.parameters(), lr=learning_rate)

def accuracy(y_pred, y_test):
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_pred.size(0))
    #acc = (np.argmax(y_pred, axis=1) == y_test).sum().float() / float(y_pred.size(0))
    return acc

for epoch in range(n_epochs):

    model.train()

    train_steps = 0
    total_train_loss = 0
    total_train_acc = 0

    with tqdm(total=len(mnli_dataloader), desc="Training MNLI ---> ") as progbar:
        for batch in mnli_dataloader:
            batch = {k: v for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits

            acc = accuracy(logits, batch['labels'])
            total_train_acc += acc

            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()

            train_steps += 1

            optimizer.step()
            optimizer.zero_grad()



            progbar.update(1)
            progbar.set_postfix_str(f"Loss: {total_train_loss / train_steps} ")

    train_acc = total_train_acc / len(mnli_dataloader)
    train_loss = total_train_loss / len(mnli_dataloader)

    model.eval()

    print(f'Epoch {epoch + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} ')

    val_steps = 0
    total_acc = 0

    with tqdm(total=len(matched_dataloader), desc="Validation MNLI matched ---> ") as progbar:
        for batch in matched_dataloader:
            batch = {k: v for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                val_loss = outputs.loss

                logits = outputs.logits
                predictions = logits #torch.argmax(logits, dim=-1)

                val_steps += 1

                acc = accuracy(predictions, batch['labels'])
                total_acc += acc

                progbar.update(1)
                #progbar.set_postfix_str(f"Accuracy: {total_acc / val_steps} ")

    val_acc = total_acc / val_steps
    print(f'Epoch {epoch + 1}: MNLI Matched val_acc: {val_acc:.4f}')

    val_steps = 0
    total_acc = 0

    with tqdm(total=len(mismatched_dataloader), desc="Validation MNLI mismatched ---> ") as progbar:
        for batch in mismatched_dataloader:
            batch = {k: v for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                val_loss = outputs.loss

                logits = outputs.logits
                predictions = logits #torch.argmax(logits, dim=-1)

                val_steps += 1

                acc = accuracy(predictions, batch['labels'])
                total_acc += acc

                progbar.update(1)
                progbar.set_postfix_str(f"Accuracy: {total_acc / val_steps} ")

    val_acc = total_acc / val_steps
    print(f'Epoch {epoch + 1}: MNLI Mismatched val_acc: {val_acc:.4f}')

    val_steps = 0
    total_acc = 0
'''
    with tqdm(total=len(diagnostics_dataloader), desc="Validation Diagnostics ---> ") as progbar:
        for batch in diagnostics_dataloader:
            batch = {k: v for k, v in batch.items()}
            print(batch)
            with torch.no_grad():
                outputs = model(**batch)
                val_loss = outputs.loss

                logits = outputs.logits
                predictions = logits #torch.argmax(logits, dim=-1)

                val_steps += 1

                acc = accuracy(predictions, batch['labels'])
                total_acc += acc
                
                progbar.update(1)
                progbar.set_postfix_str(f"Loss: {total_acc / val_steps} ")

    val_acc = total_acc / val_steps
    print(f'Epoch {epoch + 1}: MNLI Diagnostics val_acc: {val_acc:.4f}')
'''
