import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nltk
from nltk.corpus import stopwords


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

train, test = pd.read_csv('train_1.csv'), pd.read_csv('dev_1.csv')
x_train, y_train = train['sentence'], train['label']
x_test, y_test = test['sentence'], test['label']


# tokenize the words in a dataframe (must loop this for each cell in a column)
def tokenize_words(df_col):
    tokens = nltk.word_tokenize(df_col)

    tokens_no_punc = [word for word in tokens if word.isalpha()]
    normalize = [word.lower() for word in tokens_no_punc]
    stop_words = set(stopwords.words('english'))
    rem_stop_words = [word for word in normalize if word not in stop_words]

    return rem_stop_words


#print(x_train)
#print(type(x_train))
for i in range(len(train)):
    train.iloc[i, 0] = tokenize_words(train.iloc[i, 0])

print(train.head(20))
