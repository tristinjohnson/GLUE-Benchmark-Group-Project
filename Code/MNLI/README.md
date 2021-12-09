# Multi-Genre Natural Language Inference (MNLI)

This directory contains one Python file for both MNLI Matched and MNLI Mismatched tasks.

Here are the steps taken:
1. A BERT model is trained on the MNLI train dataset. Note that this dataset had to be greatly reduced for computational purposes.
2. The trained model is applied to the MNLI Matched validation set, and performance metrics are obtained.
3. The trained model is applied to the MNLI Mismatched validation set, and performance metrics are obtained.

An explanation of the MNLI tasks can be found [here](https://cims.nyu.edu/~sbowman/multinli/). 