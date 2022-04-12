# Bayer Home Assignment

## Instructions

To run the code one should [install poetry](https://python-poetry.org/docs/#installation).

After poetry installation, from the root directory of the repository one should run the following command:

```bash
poetry shell
```

That command will create a virtual environment. To install all the code dependencies you should run the following:

```bash
poetry install
```

## Hypotheses

### Logistic regression on top of the tf-idf features

To train the Logistic Regression pipeline you should run the script `tfidf_logreg.py` like below:

```bash
python tfidf_logreg.py [PATH_TO_DATASET]
```

### Fine-tuning MedBERT on downstream classification task
To fine-tune MedBERT LM you should firstly download weights:

After the weights downloaded, you can train the pipeline with the following command:

```bash
python finetune_bert.py [PATH_TO_DATASET] --path_to_bert [PATH_TO_PRETRAINED_BERT_LM]
```

### Fine-tuning MedBERT on next sentence classification task with auxiliary sentence
To fine-tune MedBERT LM you should firstly download weights:

After weights downloaded, you can train the pipeline with the following command:
```bash
python bert_auxilary_sentence.py [PATH_TO_DATASET] --path_to_bert [PATH_TO_PRETRAINED_BERT_LM]
```

The idea of the method is to transform initial text classification task to sentence pair classification via constructing auxiliary sentence.
One can assign an auxiliary sentence to each class. Then, we solve a binary classification task leveraging semantic of the classes themselves.


# Links

* [microsoft / BiomedNLP-PubMedBERT-base-uncased-abstract](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
* [Utilizing BERT for Aspect-Based Sentiment Analysis
via Constructing Auxiliary Sentence](https://arxiv.org/abs/1903.09588)

[//]: # (## Literature)

[//]: # (https://paperswithcode.com/paper/beyond-black-white-leveraging-annotator)

[//]: # ()
[//]: # (https://aclanthology.org/2021.naacl-main.204/)

[//]: # ()
[//]: # (https://resources.unbabel.com/blog/translation-ambiguity)

[//]: # ()
[//]: # (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00449/109286)

[//]: # ()
[//]: # (https://github.com/yandex-research/shifts)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (poetry add lime)