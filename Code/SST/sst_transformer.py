"""
Tristin Johnson
GLUE Dataset - Stanford Sentiment Treebank (SST)
DATS 6450 - NLP
December 9th, 2021
"""
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd

"""# num of epochs
epochs = 2

# specify glue task and get metric
task = "sst2"
metric = load_metric("glue", task)

# training and validation data
train = load_dataset("glue", name=task, split='train[0:6734]')  # train[0:6734]
validation = load_dataset("glue", name=task, split='validation')

# model checkpoint and pretrained tokenizer
checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(checkpoint, use_fast=True)


# function to tokenize the sentence in dataset
def tokenize(data):
    return tokenizer(data["sentence"], truncation=True)


# encode the training/validation data
encoded_train = train.map(tokenize, batched=True)
encoded_validation = validation.map(tokenize, batched=True)

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
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=0.0005)

# total number of training steps, and learning rate scheduler
num_training_steps = epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# train on GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Device: ', device)
model.to(device)

# train the model
for epoch in range(epochs):

    train_steps, val_steps = 0, 0

    model.train()

    with tqdm(total=len(train_dataloader), desc=f"Training: Epoch {epoch}--> ") as pbar:
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
"""

# define model parameters
batch_size = 32
num_labels = 2
max_len = 512
learning_rate = 0.001
num_epochs = 5


# tokenize dataset: get premise & entailment ids, mask ids, and token ids
# return tokenized and encoded dataset
def tokenizer_custom_dataset(tokenizer, df):
    token_ids, mask_ids, seg_ids, y = [], [], [], []

    premise_list = df['sentence'].to_list()
    labels = df['label'].to_list()

    for premise in premise_list:
        premise_ids = tokenizer.encode(premise, add_special_tokens=False)
        token_pairs = [tokenizer.cls_token_id] + premise_ids + [tokenizer.sep_token_id]

        premise_len = len(premise_ids)

        attention_mask_ids = torch.tensor([0] * (premise_len + 2))

        token_ids.append(torch.tensor(token_pairs))
        mask_ids.append(attention_mask_ids)

    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    labels = torch.tensor(labels)

    dataset = TensorDataset(token_ids, mask_ids, labels)

    return dataset


# get training and validation data loaders
def get_data_loader(train_ds, val_ds):
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size=batch_size)

    return train_loader, val_loader


# func to calculate accuracy
def rte_acc(pred, labels):
    acc = (torch.log_softmax(pred, dim=1).argmax(dim=1) == labels).sum().float() / float(pred.size(0))

    return acc


# create model definition
def model_definition():
    # define model
    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

    # define optimizer, scheduler, criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=0, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    return model, optimizer, scheduler, criterion


# train and test the model: save the best model
def train_and_test(train_loader, val_loader):
    model, optimizer, scheduler, criterion = model_definition()

    model.to(device)

    model_best_acc = 0

    for epoch in range(num_epochs):
        train_loss, train_steps = 0, 0
        corr_train_pred, total_train_pred = 0, 0

        model.train()

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}: ') as pbar:
            for idx, (token_ids, mask_ids, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                token_ids = token_ids.to(device)
                mask_ids = mask_ids.to(device)
                labels = labels.to(device)

                _, output = model(token_ids,
                                  attention_mask=mask_ids,
                                  labels=labels).values()

                loss = criterion(output, labels)
                train_loss += loss.item()
                train_steps += 1

                _, pred = torch.max(output, 1)

                corr_train_pred += (pred == labels).sum().item()  # change to output on 109
                total_train_pred += pred.shape[0]

                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix_str(f'Loss: {train_loss / train_steps:0.5f}, '
                                     f'Acc: {corr_train_pred / total_train_pred:0.5f}')

        val_loss, val_steps = 0, 0
        corr_val_pred, total_val_pred = 0, 0

        model.eval()

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f'Epoch {epoch}: ') as pbar:
                for idx, (token_ids, mask_ids, labels) in enumerate(val_loader):
                    optimizer.zero_grad()

                    token_ids = token_ids.to(device)
                    mask_ids = mask_ids.to(device)
                    labels = labels.to(device)

                    _, output = model(token_ids,
                                      attention_mask=mask_ids,
                                      labels=labels).values()

                    loss = criterion(output, labels)
                    val_loss += loss.item()
                    val_steps += 1

                    _, pred = torch.max(output, 1)

                    corr_val_pred += (pred == labels).sum().item()  # change to output on 109
                    total_val_pred += pred.shape[0]

                    pbar.update(1)
                    pbar.set_postfix_str(f'Loss: {val_loss / val_steps:0.5f}, '
                                         f'Acc: {corr_val_pred / total_val_pred:0.5f}')

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = corr_train_pred / total_train_pred
        print(f'\nEpoch {epoch} Training Results: Loss: {avg_train_loss:0.5f}, Acc: {avg_train_acc:0.5f}')

        avg_val_loss = val_loss / len(train_loader)
        avg_val_acc = corr_val_pred / total_val_pred
        print(f'Epoch {epoch} Validation Results: Loss: {avg_val_loss:0.5f}, Acc: {avg_val_acc:0.5f}')

        model_acc = avg_val_acc

        # if val accuracy is better than previous, save the model
        if model_acc > model_best_acc:
            torch.save(model.state_dict(), 'sst_bert_model.pt')
            print('This model has been saved as sst_bert_model.pt !')

            model_best_acc = model_acc

        print('\n' + 25 * '==' + '\n')


# main
if __name__ == '__main__':
    # use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device', device)

    # define GLUE task
    task = "sst2"
    metric = load_metric("glue", task)

    # get training and validation datasets
    train = load_dataset("glue", name=task, split='train')  # len=2490
    validation = load_dataset("glue", name=task, split='validation')  # 277

    # convert to dataframes
    train_df, val_df = pd.DataFrame(train), pd.DataFrame(validation)

    # define transformer tokenizer
    checkpoint = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint, do_lower_case=True)

    # tokenize datasets
    train_data = tokenizer_custom_dataset(tokenizer, train_df)
    val_data = tokenizer_custom_dataset(tokenizer, val_df)

    # get the train/validation dataloaders
    train_loader, val_loader = get_data_loader(train_data, val_data)

    # train the model
    train_and_test(train_loader, val_loader)

# best acc with BERT - train 0.54935 (15:27 per epoch), val 0.50917 (0:06 per epoch)
#
