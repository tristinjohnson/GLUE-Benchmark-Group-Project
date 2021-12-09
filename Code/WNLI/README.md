# Winograd Natural Language Inference (WNLI)

Here is the completion of the GLUE task, the Winograd Natural Language Inference. This originated from the Winograd Schema Challenge, which is a reading comprehension task in which a system must read a sentence with a pronoun and select the referent of that pronoun from a list of choices. However, the GLUE benchmark modifies the goal of this challenge. Instead of a model reading a sentence with a pronoun and choosing a referent of that pronoun from a set of choices, the goal is now to predict if a given sentence with a substituted pronoun can be inferred from the original sentence. Because of this change, the task is now called Winograd Natural Language Interference (WNLI). The dataset contains over 800 examples, in the form of two sentences and a label. Here, each example is evaluated separately so that there is no systematic correspondence between the model’s score.


In this directory, are 3 Python files: 


### 1. Transformer Base Model

This file is a quick and easy implementation of using the two transformer: ALBERT and ELECTRA. You have the ability to run both of these transformers, and recieve an Accuracy score as the testing score. Below, is how to run this file using either transformer:

    # using ELECTRA
    python3 wnli_base_model.py --model electra
    
    # using ALBERT
    python3 wnli_base_model.py --model albert
    
    
### 2. Custom Transformer Model

This file is a custom implementation of applying both ALBERT and ELECTRA using other preprocessing methods and PyTorch for training/testing. See below on how to run the code using either transformer:

    # using ELECTRA
    python3 wnli_custom_transformer.py --model electra
    
    # using ALBERT
    python3 wnli_custom_transformer.py --model albert
