from datasets import load_dataset, load_metric
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

#nltk.download()

# define model parameters
batch_size = 32
num_epochs = 25

# get stanford sentiment treebank dataset
task = "sst2"
metric = load_metric("glue", task)

# load training and validation set
train = load_dataset("glue", name=task, split='train[0:67008]')
validation = load_dataset("glue", name=task, split='validation[0:864]')

# convert train/validation to pandas df
train_df, validation_df = pd.DataFrame(train), pd.DataFrame(validation)

# convert x and y to pandas df
x_train, y_train = train_df['sentence'].values, train_df['label'].values
x_test, y_test = validation_df['sentence'].values, validation_df['label'].values

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', device)


# function to remove special characters from data
def preprocess_data(sent):
    # remove non-word chars
    sent = re.sub(r'[^\w\s]', '', sent)

    # remove multiple whitespaces
    sent = re.sub(r'\s+', '', sent)

    # remove digits
    sent = re.sub(r'\d', '', sent)

    return sent


# function to tokenize the dataset and one-hot encode data
def tokenize_and_onehot_encode(x_train, x_test):
    words = []

    # remove stopwords and convert data to lowercase
    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for w in sent.lower().split():
            w = preprocess_data(w)
            if w not in stop_words and w != '':
                words.append(w.lower())

    # define corpus and get first 1500 most common words
    corpus = Counter(words)
    corpus = sorted(corpus, key=corpus.get, reverse=True)[:1500]

    # define one_hot encoded dictionary
    one_hot_dict = {w: i+1 for i, w in enumerate(corpus)}

    new_train, new_test = [], []
    for sent in x_train:
        new_train.append([one_hot_dict[preprocess_data(w)] for w in sent.lower().split() if preprocess_data(w) in one_hot_dict.keys()])

    for sent in x_test:
        new_test.append([one_hot_dict[preprocess_data(w)] for w in sent.lower().split() if preprocess_data(w) in one_hot_dict.keys()])

    return np.array(new_train), np.array(new_test), one_hot_dict


x_train, x_test, vocab = tokenize_and_onehot_encode(x_train, x_test)


def padding(sents, sequence_len):
    features = np.zeros((len(sents), sequence_len), dtype=int)

    for i, review in enumerate(sents):
        if len(review) != 0:
            features[i, -len(review):] = np.array(review)[:sequence_len]

    return features


x_train_pad, x_test_pad = padding(x_train, 500), padding(x_test, 500)

train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
validation_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
validation_loader = DataLoader(validation_data, shuffle=True, batch_size=batch_size)

data_iterator = iter(train_loader)
sample_x, sample_y = data_iterator.next()

"""print('input size: ', sample_x.size())
print('input x: ', sample_x)
print('input y: ', sample_y)"""


class SentimentAnalysisLSTM(nn.Module):
    def __init__(self, num_layers, vocab_size, hidden_dim, embedding_dim):
        super(SentimentAnalysisLSTM, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=num_layers, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.linear = nn.Linear(self.hidden_dim, output_dim)

        self.act = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        output = self.dropout(lstm_out)

        output = self.linear(output)

        act_output = self.act(output)

        act_output = act_output.view(batch_size, -1)

        act_output = act_output[:, -1]

        return act_output, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)

        hidden = (h0, c0)

        return hidden


num_layers = 2
output_dim = 1
vocab_size = len(vocab) + 1
embedding_dim = 64
hidden_dim = 256

model = SentimentAnalysisLSTM(num_layers, vocab_size, hidden_dim, embedding_dim)
model.to(device)

print(model)

lr = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def accuracy(pred, labels):
    pred = torch.round(pred.squeeze())

    return torch.sum(pred == labels.squeeze()).item()


clip = 5
val_best_acc = 0

for epoch in range(num_epochs):
    train_loss, steps_train, train_acc = 0, 0, 0

    model.train()

    h = model.init_hidden(32)

    with tqdm(total=len(train_loader), desc='Training -->') as pbar:
        for x_data, x_target in train_loader:

            x_data, x_target = x_data.to(device), x_target.to(device)

            h = tuple([each.data for each in h])

            optimizer.zero_grad()
            output, h = model(x_data, h)
            loss = criterion(output.squeeze(), x_target.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            steps_train += 1

            acc = accuracy(output, x_target)
            train_acc += acc

            nn.utils.clip_grad_norm(model.parameters(), clip)

            pbar.update(1)
            pbar.set_postfix_str(f'Loss: {train_loss / steps_train:0.5f} '
                                 f'-> Acc: {train_acc / len(train_loader):0.5f}')

    avg_loss_train = train_loss / len(train_loader)
    acc_train = train_acc / len(train_loader)
    print(f'Training: Epoch {epoch} -> Loss: {avg_loss_train:0.5f} -> Acc: {acc_train:0.5f}')

    val_h = model.init_hidden(batch_size)

    model.eval()

    val_loss, steps_val, val_acc = 0, 0, 0

    with tqdm(total=len(validation_loader), desc='Validation -->') as pbar:
        for x_data, x_target in validation_loader:

            x_data, x_target = x_data.to(device), x_target.to(device)

            output, val_h = model(x_data, val_h)
            optimizer.zero_grad()
            loss = criterion(output.squeeze(), x_target.float())

            val_loss += loss.item()
            steps_val += 1

            acc = accuracy(output, x_target)
            val_acc += acc

            pbar.update(1)
            pbar.set_postfix_str(f'Loss: {val_loss / steps_val:0.5f} '
                                 f'-> Acc: {val_acc / len(validation_loader)}')

    avg_loss_val = val_loss / len(validation_loader)
    acc_val = val_acc / len(validation_loader)
    print(f'Validation: Epoch {epoch} -> Loss: {avg_loss_val:0.5f} -> Acc: {acc_val:0.5f}')

    if acc_val > val_best_acc:
        torch.save(model.state_dict(), 'best_model_lstm.pt')
        print('The model has been saved!')
        val_best_acc = acc_val

    print('\n' + 25*'==' + '\n')

