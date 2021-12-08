"""
Tristin Johnson
GLUE Dataset - Stanford Sentiment Treebank (SST)
DATS 6450 - NLP
December 9th, 2021
"""
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd
import argparse


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

    sent_list = df['sentence'].to_list()
    labels = df['label'].to_list()

    for sent in sent_list:
        sent_ids = tokenizer.encode(sent, add_special_tokens=False)
        token_pairs = [tokenizer.cls_token_id] + sent_ids + [tokenizer.sep_token_id]

        premise_len = len(sent_ids)

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
def model_definition(transformer):
    # define model
    model = transformer.from_pretrained(checkpoint, num_labels=num_labels)

    # define optimizer, scheduler, criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=0, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    return model, optimizer, scheduler, criterion


# train and test the model: save the best model
def train_and_test(train_loader, val_loader, transformer):
    model, optimizer, scheduler, criterion = model_definition(transformer)

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
            torch.save(model.state_dict(), 'sst_model.pt')
            print('This model has been saved as sst_bert_model.pt !')

            model_best_acc = model_acc

        print('\n' + 25 * '==' + '\n')


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert', type=str, help="select any of the model: ['bert', 'electra']")
    args = parser.parse_args()

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
    if args.model == 'bert':
        checkpoint = 'bert-base-uncased'
        tokenizer = BertTokenizerFast.from_pretrained(checkpoint, do_lower_case=True)
        transformer = BertForSequenceClassification
        print('\nUsing BERT Transformer!\n')

    elif args.model == 'electra':
        checkpoint = "google/electra-small-discriminator"
        tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator", do_lower_case=True)
        transformer = ElectraForSequenceClassification
        print('\nUsing Electra Transformer!\n')

    elif args.model == 'mobile_bert':
        checkpoint = "google/mobilebert-uncased"
        tokenizer = MobileBertTokenizer.from_pretrained(checkpoint, do_lower_case=True)
        transformer = MobileBertForSequenceClassification
        print('\nUsing MobileBert Transformer!\n')


    # tokenize datasets
    train_data = tokenizer_custom_dataset(tokenizer, train_df)
    val_data = tokenizer_custom_dataset(tokenizer, val_df)

    # get the train/validation dataloaders
    train_loader, val_loader = get_data_loader(train_data, val_data)

    # train the model
    train_and_test(train_loader, val_loader, transformer)

# best acc with BERT - train 0.54935 (15:27 per epoch), val 0.50917 (0:06 per epoch)
# best acc with Electra - train ( per epoch), val ( per epoch)
