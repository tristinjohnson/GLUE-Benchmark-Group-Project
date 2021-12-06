from datasets import load_dataset, load_metric
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer, get_linear_schedule_with_warmup, get_scheduler
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm
import pandas as pd
import random

num_epochs = 2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

task = 'cola'
metric = load_metric("glue", task)

# training and validation
train = load_dataset("glue", name=task, split='train')
validation = load_dataset("glue", name=task, split='validation')

# convert to pandas df
data = pd.concat([pd.DataFrame(train), pd.DataFrame(validation)])

# get sentences and labels
sentences = data.sentence.values
labels = data.label.values

# define tokenizer
checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

print('original: ', sentences[0])
print('Tokenized: ', tokenizer.tokenize(sentences[0]))
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

max_len = 0  # max length is 47

"""for sent in sentences:

    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)"""

input_ids, attention_masks = [], []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(sent,
                                         add_special_tokens=True,
                                         max_length=64,
                                         padding='max_length',
                                         return_attention_mask=True,
                                         truncation=True,
                                         return_tensors='pt')

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

print('original: ', sentences[0])
print('token IDs: ', input_ids[0])

print(input_ids.shape)
print(attention_masks.shape)
print(labels.shape)

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.9 * len(dataset))
validation_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, validation_size])

batch_size = 32
num_epochs = 2

train_dataloader = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=batch_size)
val_dataloader = DataLoader(val_ds, sampler=SequentialSampler(val_ds), batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained(checkpoint,
                                                      num_labels=2,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def accuracy(predictions, labels):
    pred = np.argmax(predictions, axis=1).flatten()
    label = labels.flatten()

    return np.sum(pred == label) / len(label)


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

training_stats = []

for epoch in range(num_epochs):
    train_loss, train_steps, train_acc = 0, 0, 0

    model.train()

    with tqdm(total=len(train_dataloader), desc='Training --> ') as pbar:

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            train_loss += loss.item()
            train_steps += 1

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            train_acc += accuracy(logits, label_ids)

            pbar.update()
            pbar.set_postfix_str(f'Loss: {train_loss / train_steps} '
                                 f'-> Acc: {train_acc / train_steps}')

    avg_train_loss = train_loss / len(train_dataloader)
    avg_train_acc = train_acc / len(train_dataloader)

    print(f'\nEpoch {epoch}: Accuracy: {avg_train_acc:0.5f} -> Loss: {avg_train_loss:0.5f}\n')

    training_stats.append({'epoch': epoch,
                           'training_loss': avg_train_loss,
                           'training_acc': avg_train_acc})











"""for epoch in range(num_epochs):

    model.train()

    train_loss, train_acc, train_steps = 0, 0, 0

    with tqdm(total=len(train_dataloader), desc='Training --> ') as pbar:

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f'Loss: {train_loss / train_steps}')

    avg_train_loss = train_loss / len(train_dataloader)
    avg_train_acc = train_acc / len(train_dataloader)

    print(f'Training: Loss: {avg_train_loss} --> Acc: {avg_train_acc}')
"""
