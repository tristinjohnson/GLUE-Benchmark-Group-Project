from datasets import load_dataset, load_metric
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import tqdm
import pandas as pd
import random


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

task = 'cola'
metric = load_metric("glue", task)

# training and validation
train = load_dataset("glue", name=task, split='train')
validation = load_dataset("glue", name=task, split='validation')

# convert to pandas df
data = pd.concat([pd.DataFrame(train), pd.DataFrame(validation)])

sentences = data.sentence.values
labels = data.label.values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#print('original: ', sentences[0])
#print('Tokenized: ', tokenizer.tokenize(sentences[0]))
#print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))


input_ids, attention_masks = [], []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(sent,
                                         padding='max_length',
                                         max_length=64,
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

train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_ds, shuffle=True, batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=2,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

total_steps = len(train_dataloader) * num_epochs

model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def accuracy(predictions, labels):
    pred = np.argmax(predictions, axis=1).flatten()
    label = labels.flatten()

    return np.sum(pred == label) / len(label)


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

for epoch in range(num_epochs):

    model.train()

    train_loss, train_acc, train_steps = 0, 0, 0

    with tqdm(total=len(train_dataloader), desc='Training --> ') as pbar:

        for step, batch in enumerate(train_dataloader):
            """b_input_ids = batch[0].to(device)
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
            scheduler.step()"""

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            train_loss = outputs.loss
            train_loss.backward()

            train_steps += 1

            optimizer.step()
            optimizer.zero_grad()

            pbar.update(1)
            pbar.set_postfix_str(f'Loss: {train_loss / train_steps}')

    avg_train_loss = train_loss / len(train_dataloader)
    avg_train_acc = train_acc / len(train_dataloader)

    print(f'Training: Loss: {avg_train_loss} --> Acc: {avg_train_acc}')

