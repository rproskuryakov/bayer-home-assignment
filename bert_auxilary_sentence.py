import argparse
import logging
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertForNextSentencePrediction
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

import src.callbacks
import src.datasets
import src.metrics
import src.trainer
import src.utils


logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT model for sentiment classification task on sentence and auxilary sentence."
    )
    parser.add_argument("dataset_path", type=pathlib.Path)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--full_finetuning", type=bool, default=False)
    parser.add_argument(
        "--path_to_bert",
        type=str,
        default="./models/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument(
        "--path_to_checkpoints", type=str, default="./models/bert_pair_clf/checkpoints/"
    )
    args = parser.parse_args()

    RANDOM_SEED = 6042022
    src.utils.set_seed(RANDOM_SEED)

    tokenizer = BertTokenizer.from_pretrained(args.path_to_bert, do_lower_case=False)
    logger.info("Tokenizer is loaded.")

    LABEL_TO_STATEMENT = {
        "negative": "Trial results are unsuccessful.",
        "neutral": "Trial results are unclear.",
        "positive": "Trial results are successful",
    }

    ## READ DATASET
    train_data, test_data, train_target, test_target = src.utils.get_train_test_split(
        args.dataset_path, RANDOM_SEED
    )

    # solving aspect-category sentence classification
    train_dataset = src.datasets.SentimentAnalysisPairClassificationTask(
        train_data.tolist(),
        train_target.tolist(),
        tokenizer=tokenizer,
        label_to_statement=LABEL_TO_STATEMENT,
    )

    test_dataset = src.datasets.SentimentAnalysisPairClassificationTask(
        test_data.tolist(),
        test_target.tolist(),
        tokenizer=tokenizer,
        label_to_statement=LABEL_TO_STATEMENT,
    )
    changed_targets = np.array(
        [batch.get("labels") for batch in train_dataset]
    ).flatten()
    CLASS_WEIGHTS = compute_class_weight("balanced", classes=[0, 1], y=changed_targets)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )
    valid_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    logger.info(
        "Initializing BertForNextSentenceClassification from pretrained BERT LM"
    )
    model = BertForNextSentencePrediction.from_pretrained(
        args.path_to_bert,
        output_attentions=False,
        output_hidden_states=False,
    )

    if args.full_finetuning:
        param_optimizer = list(model.cls.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=1e-8,
    )

    # add a scheduler to linearly reduce the learning rate throughout the epochs
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * args.n_epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        logger.info("CUDA is available")
        model.cuda()

    trainer = src.trainer.Trainer(
        model=model,
        main_metric=src.metrics.F1Score(),
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader,
        device=device,
        scheduler=scheduler,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        callbacks=[
            src.callbacks.SaveCheckpoints(
                model, args.path_to_checkpoints, only_best=False, mode="max"
            ),
            src.callbacks.EarlyStopping(),
        ],
        metrics=[
            src.metrics.RecallScore(),
            src.metrics.PrecisionScore(),
        ],
        loss=nn.CrossEntropyLoss(weight=torch.FloatTensor(CLASS_WEIGHTS)),
    )
    trainer.fit()

    model.eval()

    logger.info("Starting to calculate answers on the test set.")

    predicted_labels = [
        src.utils.predict_label_for_aux_sentence_model(
            text, model, tokenizer, LABEL_TO_STATEMENT
        )
        for text in test_data.tolist()
    ]

    logger.info(
        "F1 Macro ON PROPER VALIDATION: %.2f"
        % f1_score(test_target, predicted_labels, average="macro")
    )
    logger.info(
        "MACRO Precision on untransformed dataset: %.2f"
        % precision_score(test_target, predicted_labels, average="macro")
    )
    logger.info(
        "Macro Recall on untransformed dataset: %.2f"
        % recall_score(test_target, predicted_labels, average="macro")
    )
