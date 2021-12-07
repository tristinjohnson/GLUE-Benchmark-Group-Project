from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_scheduler
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm

# Check GPU availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
# Set batch size to use for our model
batch_size = 32
# Set learning rate for our models
learning_rate = 0.0005
# Epochs for training the model
n_epochs = 5

# Load data set and metrics
# Task options are
# ["sst2", "mnli", "mnli_mismatched", "mnli_matched", "cola", "stsb", "mrpc", "qqp", "qnli", "rte", "wnli", "hans"]
# Use mnli since it has train dataset. For simplicity, pull test for matched from the same source
task = 'mnli'

metric = load_metric('glue', task)
# Get train and test as datatype <class 'datasets.arrow_dataset.Dataset'>
train = load_dataset('glue', task, split='train[0:30000]') # Full sample is 392702 but only taking 30000
test = load_dataset('glue', task, split='test_matched') # Full sample is 9796, getting all of the sample

# Load model
checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint, use_fast=True)

# Tokenize train and test
# Append the two together with specific tokens
premise = train['premise']
hypothesis = train['hypothesis']
label = train['label']

token_list = []
seq_list = []
mask_list = []
y = []

# Loop through premise and hypothesis, create tokens based on the concatenation of both
# add_special_tokens=True adds cls_token id to start (101) and sep_token_id (102) to end of each seq 1 and seq 2
for p, h in zip(premise, hypothesis):
    tok_info = tokenizer(p, h, add_special_tokens=True)
    token_id, seq_id, mask_id = tok_info['input_ids'], tok_info['token_type_ids'], tok_info['attention_mask']
    # Append token, seq info, and mask info to list as tensors
    token_list.append(torch.tensor(token_id))
    seq_list.append(torch.tensor(seq_id))
    mask_list.append(torch.tensor(mask_id))

# Turn label into target variable for our model
# Get label coded using a dictionary
label_dict = {'entailment': 0,
              'contradiction': 1,
              'neutral': 2}

for val in label:
    y.append(label_dict[val])

y = torch.tensor(y)

# Pad for model
token_list = pad_sequence(token_list, batch_first=True)
seq_list = pad_sequence(seq_list, batch_first=True)
mask_list = pad_sequence(mask_list, batch_first=True)
data_set = TensorDataset(token_list, mask_list, seq_list, y)

# Load data into batches
train_dataloader = DataLoader(data_set, shuffle=True, batch_size=batch_size)

###########################
# Repeat data steps for validation set

premise = test['premise']
hypothesis = test['hypothesis']
label = test['label']

token_list = []
seq_list = []
mask_list = []
y = []

# Loop through premise and hypothesis, create tokens based on the concatenation of both
# add_special_tokens=True adds cls_token id to start (101) and sep_token_id (102) to end of each seq 1 and seq 2
for p, h in zip(premise, hypothesis):
    tok_info = tokenizer(p, h, add_special_tokens=True)
    token_id, seq_id, mask_id = tok_info['input_ids'], tok_info['token_type_ids'], tok_info['attention_mask']
    # Append token, seq info, and mask info to list as tensors
    token_list.append(torch.tensor(token_id))
    seq_list.append(torch.tensor(seq_id))
    mask_list.append(torch.tensor(mask_id))

for val in label:
    y.append(label_dict[val])

y = torch.tensor(y)

# Pad for model
token_list = pad_sequence(token_list, batch_first=True)
seq_list = pad_sequence(seq_list, batch_first=True)
mask_list = pad_sequence(mask_list, batch_first=True)
data_set = TensorDataset(token_list, mask_list, seq_list, y)

# Load data into batches
val_dataloader = DataLoader(data_set, shuffle=True, batch_size=batch_size)

###########################


# Create model
# Model needs tokens, sequence_ids, mask_ids, and label (turned to a factor)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler to improve model performance
# This will step along with the optimizer in the model
num_training_steps = n_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

train_steps = 0
val_steps = 0

# Define function to calculate softmax among classes for prediction
def accuracy(y_pred, y_test):
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_pred.size(0))
    return acc

# Model training
for epoch in range(n_epochs):
    model.train()

    total_train_loss = 0
    total_train_acc = 0

    with tqdm(total=len(train_dataloader), desc="Training ---> ") as progbar:
        for batch_id, (token, mask, seg, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            token = token.to(device)
            mask = mask.to(device)
            seg = seg.to(device)
            labels = y.to(device)

            loss, pred = model(token, seg, mask, labels=labels).values()

            acc = accuracy(pred, labels)

            loss.backward()

            train_steps += 1
            lr_scheduler.step()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += acc.item()

            progbar.update(1)
            progbar.set_postfix(f"Loss: {total_train_loss / train_steps} ")

    train_acc = total_train_acc / len(train_dataloader)
    train_loss = total_train_loss / len(train_dataloader)

    model.eval()

    total_val_loss = 0
    total_val_acc = 0

    with tqdm(total=len(val_dataloader), desc="Training ---> ") as progbar:
        for batch_id, (token, mask, seg, y) in enumerate(val_dataloader):
            optimizer.zero_grad()
            token = token.to(device)
            mask = mask.to(device)
            seg = seg.to(device)
            labels = y.to(device)

            loss, pred = model(token, seg, mask, labels=labels).values()

            acc = accuracy(pred, labels)

            total_val_loss += loss.item()
            total_val_acc += acc.item()

            progbar.update(1)
            progbar.set_postfix(f"Loss: {total_val_loss / val_steps} ")

        val_acc = total_val_acc / len(val_dataloader)
        val_loss = total_val_loss / len(val_dataloader)

    print(f'Epoch {epoch + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')