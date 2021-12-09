# Recognizing Textual Entailment (RTE)

Here is the completion of the GLUE task, the Recognizing Textual Entailment. This dataset is a combination of 4 other RTE datasets and contains over 5,500 examples. The goal of textual entailment recognition is to train a model that takes two text fragments and sees whether the meaning of one text fragment can be entailed/inferred from the other text fragment.

In this directory, are 3 Python files: 


### 1. Transformer Base Model

This file is a quick and easy implementation of using the two transformer: ALBERT and ELECTRA. You have the ability to run both of these transformers, and recieve an Accuracy score as the testing score. Below, is how to run this file using either transformer:

    # using ELECTRA
    python3 rte_base_model.py --model electra
    
    # using ALBERT
    python3 rte_base_model.py --model albert
    
    
### 2. Custom Transformer Model

This file is a custom implementation of applying both ALBERT and ELECTRA using other preprocessing methods and PyTorch for training/testing. See below on how to run the code using either transformer:

    # using ELECTRA
    python3 rte_custom_transformer.py --model electra
    
    # using ALBERT
    python3 rte_custom_transformer.py --model albert

