from datasets import load_dataset
from transformers import pipeline

# define all of the GLUE tasks
glue_tasks = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]


def question_answering(split):
    # get name of task and define transformer pipeline
    task = glue_tasks[4]
    classifier = pipeline("question-answering")

    if split == 'validation':
        validation = load_dataset("glue", name=task, split='validation')
        question, context, label = validation['question'], validation['sentence'], validation['label']

    test = classifier(question=question[0], context=context[0])
    print(test)
    print(label[0])

    test = classifier(question=question[1], context=context[1])
    print(test)
    print(label[1])

    test = classifier(question=question[2], context=context[2])
    print(test)
    print(label[2])

    test = classifier(question=question[3], context=context[3])
    print(test)
    print(label[3])

    for i in range(len(question[0:10])):
        pred = []
        sent = classifier(question=question[i], context=context[i])
        pred.append(sent['score'])
        print(pred, ' --> ', label[i])

    # 0 == entailment ( < 0.5 ): 1 == not_entailment ( > 0.5 )


# function to perform sentiment analysis on GLUE dataset using transformers pipeline
def sentiment_analysis(split):
    # get name of task and define transformer pipeline
    task = glue_tasks[7]
    classifier = pipeline("sentiment-analysis")

    if split == 'train':
        train = load_dataset("glue", name=task, split='train')
        sentences, labels = train['sentence'], train['label']

    if split == 'validation':
        validation = load_dataset("glue", name=task, split='validation')
        sentences, labels = validation['sentence'], validation['label']

    correct_labels = 0
    for i in range(len(sentences)):
        pred = []
        sent = classifier(sentences[i])
        pred.append(sent[0]['label'])
        pred = [1 if label == 'POSITIVE' else 0 for label in pred]

        if labels[i] == pred[0]:
            correct_labels += 1

    accuracy = correct_labels / len(sentences)
    print(f'\nAccuracy: {accuracy*100:0.3f}%')


# function to select a GLUE dataset and perform the given task
def transformer_selection(dataset, split):
    if dataset == 'sst2':
        print('\nUsing pre-defined transformer for sentiment analysis on Stanford Sentiment Treebank\n')
        sentiment_analysis(split)

    if dataset == 'qnli':
        question_answering(split)


if __name__ == '__main__':
    #transformer_selection('sst2', 'validation')
    transformer_selection('qnli', 'validation')

