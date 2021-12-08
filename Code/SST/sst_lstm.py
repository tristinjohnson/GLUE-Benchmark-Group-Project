"""
Tristin Johnson
DATS 6450 - NLP
December 9th, 2021
"""
# import various required packages
from datasets import load_dataset
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')


# define model parameters
batch_size = 32
num_epochs = 20
num_layers = 2
output_dim = 1
embedding_dim = 64
hidden_dim = 256
max_len = 256


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
def tokenize_and_onehot_encode(x_train, x_test, data_split):
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
        new_train.append([one_hot_dict[preprocess_data(w)] for w in sent.lower().split()
                          if preprocess_data(w) in one_hot_dict.keys()])

    # data split is test, only return one-hot encoded test set and dictionary
    if data_split == 'test':
        return np.array(new_train), one_hot_dict

    # if split is training, return new train, new test, and dictionary
    else:
        for sent in x_test:
            new_test.append([one_hot_dict[preprocess_data(w)] for w in sent.lower().split()
                             if preprocess_data(w) in one_hot_dict.keys()])

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

        # define LSTM parameters
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # define architecture
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm1 = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                             num_layers=num_layers, batch_first=True)

        self.lstm2 = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                             num_layers=num_layers, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.linear = nn.Linear(self.hidden_dim, output_dim)

        self.act = nn.Sigmoid()

    # forward propagation
    def forward(self, x, hidden):
        batch_size = x.size(0)

        embeds = self.embedding(x)

        lstm_out, hidden1 = self.lstm1(embeds, hidden)
        lstm_out2, hidden2 = self.lstm2(embeds, hidden1)

        lstm_out2 = lstm_out2.contiguous().view(-1, self.hidden_dim)

        output = self.dropout(lstm_out2)

        output = self.linear(output)

        act_output = self.act(output)

        act_output = act_output.view(batch_size, -1)

        act_output = act_output[:, -1]

        return act_output, hidden

    # initialize hidden layer
    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)

        hidden = (h0, c0)

        return hidden


# define the model
def model_definition():
    model = SentimentAnalysisLSTM(num_layers, vocab_size, hidden_dim, embedding_dim)
    model.to(device)

    print(model)

    # define model parameters
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
        train_loss, steps_train = 0, 0
        corr_train_pred, total_train_pred = 0, 0

        model.train()

        # initialize hidden layer
        h = model.init_hidden(32)

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}: ') as pbar:
            for x_data, x_target in train_loader:

                x_data, x_target = x_data.to(device), x_target.to(device)

                # create new variables for each hidden state
                h = tuple([each.data for each in h])

                output, h = model(x_data, h)
                loss = criterion(output.squeeze(), x_target.float())

                train_loss += loss.item()
                steps_train += 1

                pred = torch.round(output.squeeze())

                corr_train_pred += (pred == x_target).sum().item()
                total_train_pred += pred.shape[0]

                nn.utils.clip_grad_norm(model.parameters(), clip)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                pbar.update(1)
                pbar.set_postfix_str(f'Loss: {train_loss / steps_train:0.5f} '
                                     f'-> Acc: {corr_train_pred / total_train_pred:0.5f}')

        # initialize validation hidden layer
        val_h = model.init_hidden(batch_size)

        model.eval()

        val_loss, steps_val = 0, 0
        corr_val_pred, total_val_pred = 0, 0

        with torch.no_grad():
            with tqdm(total=len(validation_loader), desc=f'Epoch {epoch}: ') as pbar:
                for x_data, x_target in validation_loader:

                    x_data, x_target = x_data.to(device), x_target.to(device)

                    output, val_h = model(x_data, val_h)
                    optimizer.zero_grad()
                    loss = criterion(output.squeeze(), x_target.float())

                    val_loss += loss.item()
                    steps_val += 1

                    pred = torch.round(output.squeeze())

                    corr_val_pred += (pred == x_target).sum().item()
                    total_val_pred += pred.shape[0]

                    pbar.update(1)
                    pbar.set_postfix_str(f'Loss: {val_loss / steps_val:0.5f} '
                                         f'-> Acc: {corr_val_pred / total_val_pred}')

        # get metrics from training
        avg_loss_train = train_loss / steps_train
        av_acc_train = corr_train_pred / total_train_pred
        print(f'Training: Epoch {epoch} -> Loss: {avg_loss_train:0.5f} -> Acc: {av_acc_train:0.5f}')

        # output validation metrics
        avg_loss_val = val_loss / steps_val
        avg_acc_val = corr_val_pred / total_val_pred
        print(f'Validation: Epoch {epoch} -> Loss: {avg_loss_val:0.5f} -> Acc: {avg_acc_val:0.5f}')

        # save the best model
        if avg_acc_val > val_best_acc:
            torch.save(model.state_dict(), 'sst_lstm_model.pt')
            print('The model has been saved!')
            val_best_acc = avg_acc_val

        print('\n' + 25*'==' + '\n')


# test the model
def test_model(test_loader, sentences):
    # load the model from training/validation
    model, criterion, optimizer, scheduler = model_definition()
    model.load_state_dict(torch.load('sst_lstm_model.pt', map_location=device))

    test_loss, steps_test = 0, 0
    test_predictions = []

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

                pred = torch.round(output.squeeze())

                test_predictions.append(pred.detach().cpu().numpy())

                pbar.update(1)
                pbar.set_postfix_str(f'Loss: {test_loss / steps_test:0.5f}')

    # output testing metrics (loss)
    avg_loss_test = test_loss / len(test_loader)
    print(f'Test Set -> Loss: {avg_loss_test:0.5f}')

    # write predictions to an excel file (no labels on test set)
    results_df = pd.DataFrame(sentences, columns=['Sentences'])
    test_pred = np.concatenate(test_predictions)
    results_df['model_predictions'] = test_pred

    results_df.to_excel('sst_lstm_model_submission.xlsx', index=False)

    print('\n Your model predictions have been saved to an excel file in this directory --> sst_lstm_model_results.xlsx')


# load the training data, perform preprocessing, and return training/validation DataLoaders and vocab
def load_training_data(task):
    # load training and validation set
    train = load_dataset("glue", name=task, split='train')  # 67328
    validation = load_dataset("glue", name=task, split='validation')  # 864

    # convert train/validation to pandas df
    train_df, validation_df = pd.DataFrame(train), pd.DataFrame(validation)

    # convert x and y to pandas df
    x_train, y_train = train_df['sentence'].values, train_df['label'].values
    x_test, y_test = validation_df['sentence'].values, validation_df['label'].values

    # one hot encode train and validation, and get the vocab
    x_train, x_test, vocab = tokenize_and_onehot_encode(x_train, x_test, 'train')
    vocab_size = len(vocab) + 1

    # add padding to train and test
    x_train_pad, x_test_pad = padding(x_train, max_len), padding(x_test, max_len)

    # create Tensors
    train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
    validation_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

    # input Tensors into DataLoader
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    validation_loader = DataLoader(validation_data, shuffle=True, batch_size=batch_size, drop_last=True)

    return train_loader, validation_loader, vocab_size


# load in testing data, preprocess dataset, return test DataLoader, sentences for results.xlsx, and vocab
def load_testing_data(task):
    # load testing data
    test = load_dataset("glue", name=task, split='test')
    test_df = pd.DataFrame(test)

    # define x and y test and sentences for results
    # for this test set, there are no labeled values. the variables 'y_test' is only
    # defined as a variable to pass into the 'tokenize_and_onehot_encode' function,
    # and if you look at this function, y_test variable does nothing
    # with one-hot encoding the test set
    x_test, y_test = test_df['sentence'].values, test_df['label'].values
    sentences = test_df['sentence'][0:1792].values

    # tokenzie and one-hot encode test set and get the vocab
    x_test, vocab = tokenize_and_onehot_encode(x_test, y_test, 'test')
    vocab_size = len(vocab) + 1

    # add padding to x_Test
    x_test_pad = padding(x_test, max_len)

    # convert test set to a Tensor Dataset
    test_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

    # get the Test DataLoader
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)

    return test_loader, sentences, vocab_size


if __name__ == '__main__':
    # define task name --> SST
    task = "sst2"

    # use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device: ', device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train_and_validate', type=str, help="the split you would like to apply")
    args = parser.parse_args()

    # if running training/validation data, run the training loop to generate model
    if args.split == 'train_and_validate':
        train_loader, val_loader, vocab_size = load_training_data(task)
        train_and_validate(train_loader, val_loader)

    # if running test, run test_model after running train_and_validate and generating model
    elif args.split == 'test':
        test_loader, sentences, vocab_size = load_testing_data(task)
        test_model(test_loader, sentences)

# best acc with LSTM - train 0.88419 (0:52 per epoch), val 0.77315 (0:02 per epoch)
