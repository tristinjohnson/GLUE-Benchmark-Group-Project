"""
Tristin Johnson
DATS 6450 - NLP
December 9th, 2021
"""
# import various required packages
from datasets import load_dataset, load_metric
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# define model parameters
batch_size = 32
num_epochs = 25
num_layers = 2
output_dim = 1
embedding_dim = 64
hidden_dim = 256


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


# add padding to the features
def padding(sents, sequence_len):
    features = np.zeros((len(sents), sequence_len), dtype=int)

    for i, review in enumerate(sents):
        if len(review) != 0:
            features[i, -len(review):] = np.array(review)[:sequence_len]

    return features


# class to apply Sentiment Analysis using LSTM
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


# function to calculate accuracy
def accuracy(pred, labels):
    pred = torch.round(pred.squeeze())

    return torch.sum(pred == labels.squeeze()).item()


# define the model
def model_definition():
    model = SentimentAnalysisLSTM(num_layers, vocab_size, hidden_dim, embedding_dim)
    model.to(device)

    print(model)

    lr = 0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=0, verbose=True)

    return model, criterion, optimizer, scheduler


# train and validate the model
def train_and_validate(train_loader, validation_loader):
    model, criterion, optimizer, scheduler = model_definition()

    clip = 5
    val_best_acc = 0

    for epoch in range(num_epochs):
        train_loss, steps_train, train_acc = 0, 0, 0

        model.train()

        # initialize hidden layer
        h = model.init_hidden(32)

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}: ') as pbar:
            for x_data, x_target in train_loader:

                x_data, x_target = x_data.to(device), x_target.to(device)

                h = tuple([each.data for each in h])

                output, h = model(x_data, h)
                loss = criterion(output.squeeze(), x_target.float())

                train_loss += loss.item()
                steps_train += 1

                acc = accuracy(output, x_target)
                train_acc += acc

                nn.utils.clip_grad_norm(model.parameters(), clip)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                pbar.update(1)
                pbar.set_postfix_str(f'Loss: {train_loss / steps_train:0.5f} '
                                     f'-> Acc: {train_acc / len(train_loader):0.5f}')

        # get metrics from training
        avg_loss_train = train_loss / len(train_loader)
        acc_train = train_acc / len(train_loader)
        print(f'Training: Epoch {epoch} -> Loss: {avg_loss_train:0.5f} -> Acc: {acc_train:0.5f}')

        # initialize validation hidden layer
        val_h = model.init_hidden(batch_size)

        model.eval()

        val_loss, steps_val, val_acc = 0, 0, 0

        with torch.no_grad():
            with tqdm(total=len(validation_loader), desc=f'Epoch {epoch}: ') as pbar:
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

        # output validation metrics
        avg_loss_val = val_loss / len(validation_loader)
        acc_val = val_acc / len(validation_loader)
        print(f'Validation: Epoch {epoch} -> Loss: {avg_loss_val:0.5f} -> Acc: {acc_val:0.5f}')

        # save the best model
        if acc_val > val_best_acc:
            torch.save(model.state_dict(), 'best_model_lstm.pt')
            print('The model has been saved!')
            val_best_acc = acc_val

        print('\n' + 25*'==' + '\n')


# test the model
def test_model(test_loader):
    model, criterion, optimizer, scheduler = model_definition()
    model.load_state_dict(torch.load('best_model_lstm.pt', map_location=device))

    test_loss, steps_test, test_acc = 0, 0, 0

    # initialize hidden layer
    test_h = model.init_hidden(batch_size)

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Testing --> ') as pbar:
            for x_data, x_target in test_loader:
                x_data, x_target = x_data.to(device), x_target.to(device)

                output, test_h = model(x_data, test_h)

                loss = criterion(output.squeeze(), x_target.float())

                test_loss += loss.item()
                steps_test += 1

                acc = accuracy(output, x_target)
                test_acc += acc

                pbar.update(1)
                pbar.set_postfix_str(f'Loss: {test_loss / steps_test:0.5f} '
                                     f'-> Acc: {test_acc / len(test_loader)}')

    # output validation metrics
    avg_loss_test = test_loss / len(validation_loader)
    acc_test = test_acc / len(validation_loader)
    print(f'Test Set -> Loss: {avg_loss_test:0.5f} -> Acc: {acc_test:0.5f}')


if __name__ == '__main__':
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

    # use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device: ', device)

    # one hot encode train and validation, and get the vocab
    x_train, x_test, vocab = tokenize_and_onehot_encode(x_train, x_test)

    # add padding to train and test
    x_train_pad, x_test_pad = padding(x_train, 500), padding(x_test, 500)

    # create Tensors
    train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
    validation_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

    # input Tensors into DataLoader
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(validation_data, shuffle=True, batch_size=batch_size)

    # define data iterator and get vocab size
    data_iterator = iter(train_loader)
    sample_x, sample_y = data_iterator.next()
    vocab_size = len(vocab) + 1

    # train and validate
    train_and_validate(train_loader, validation_loader)

    # TODO: define the testing function to test models performance
    """test = load_dataset("glue", name=task, split='test')
    test_df = pd.DataFrame(test)

    x_test, y_test = test_df['sentence'].values, test_df['label'].values

    x_train, x_test, vocab = tokenize_and_onehot_encode(x_train, x_test)"""


