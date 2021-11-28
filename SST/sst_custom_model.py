from datasets import load_dataset, load_metric
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from collections import Counter


# define model parameters
batch_size = 64
num_epochs = 2

# get stanford sentiment treebank dataset
task = "sst2"
metric = load_metric("glue", task)

# load training and validation set
train = load_dataset("glue", name=task, split='train[0:6734]')
validation = load_dataset("glue", name=task, split='validation[0:250]')

# convert x and y to pandas df
x_train, y_train = pd.DataFrame(train['sentence']), pd.DataFrame(train['label'])
x_test, y_test = pd.DataFrame(validation['sentence']), pd.DataFrame(validation['label'])


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
    stopword = set(stopwords.words('english'))
    for sent in x_train:
        for w in sent.lower():
            w = preprocess_data(w)
            if w not in stopword and w != '':
                words.append(w)

    # define corpus and get first 1500 most common words
    corpus = Counter(words)
    corpus = sorted(corpus, reverse=True)[:1500]

    # define one_hot encoded dictionary
    one_hot_dict = {w: i+1 for i, w in enumerate(corpus)}

    new_train, new_test = [], []
    for sent in x_train:
        new_train.append([one_hot_dict[preprocess_data(w)] for w in sent.lower().split() if preprocess_data(w) in one_hot_dict.keys()])

    for sent in x_test:
        new_train.append([one_hot_dict[preprocess_data(w)] for w in sent.lower().split() if preprocess_data(w) in one_hot_dict.keys()])

    return np.array(new_train), np.array(new_test), one_hot_dict


x_train, x_test, vocab = tokenize_and_onehot_encode(x_train, x_test)
print(len(vocab))
print(x_train)
print(x_train)





