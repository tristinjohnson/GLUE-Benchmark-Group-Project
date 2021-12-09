# Stanford Sentiment Treebank (SST)

Here is the completion of the GLUE task, the Stanford Sentiment Treebank. This dataset contains over 68,000 sentences from multiple movie reviews and human annotations of each review’s sentiment. Therefore, the goal with this task is to train a model that correctly classifies the sentiment of a given movie review as either positive or negative.

In this directory, are 3 Python files: 


### 1. Transformer Base Model

This file is a quick and easy implementation of using the two transformer: ALBERT and ELECTRA. You have the ability to run both of these transformers, and recieve an Accuracy score as the testing score. Below, is how to run this file using either transformer:

    # using ELECTRA
    python3 sst_base_model.py --model electra
    
    # using ALBERT
    python3 sst_base_model.py --model albert
    
    
### 2. Custom Transformer Model

This file is a custom implementation of applying both ALBERT and ELECTRA using other preprocessing methods and PyTorch for training/testing. See below on how to run the code using either transformer:

    # using ELECTRA
    python3 sst_custom_transformer.py --model electra
    
    # using ALBERT
    python3 sst_custom_transformer.py --model albert
    
    
### 3. LSTM Model

This file contains a customized LSTM model implementation. All of the data preprocessing and training/testing was implemented using no transformers. First, you must train the model. To do so, enter the following command:

    # train, validate, and save best model
    python3 sst_lstm.py --split train_and_validate
    
After running this, you will notice that the model has saved in the same directory, titled 'sst_lstm_model.pt'. Once you have generated this model, you can run the testing split, which outputs a submissions.xlsx file labeled 'sst_lstm_model_submission.xlsx'. This file is an example of what a test submission file would look like. See below on how to run the test script:

    # make predictions by loading model generated in training
    python3 sst_lstm.py --split test
