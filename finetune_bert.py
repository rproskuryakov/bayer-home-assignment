import argparse
import logging
import pathlib
import sys
import warnings

import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertForSequenceClassification
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
        description="Fine-tune BERT model for sentiment classification task."
    )
    parser.add_argument("dataset_path", type=pathlib.Path)
    parser.add_argument("--batch_size", type=int, default=16)
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
        "--path_to_checkpoints",
        type=str,
        default="./models/bert_classifier/checkpoints/",
    )
    args = parser.parse_args()

    RANDOM_SEED = 6042022
    src.utils.set_seed(RANDOM_SEED)

    tokenizer = BertTokenizer.from_pretrained(args.path_to_bert, do_lower_case=False)
    logger.info("Tokenizer is loaded.")

    LABEL_TO_ID = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    # READ DATASET
    train_data, test_data, train_target, test_target = src.utils.get_train_test_split(
        args.dataset_path, RANDOM_SEED
    )

    # solving aspect-category sentence classification
    train_dataset = src.datasets.SentimentAnalysisExcelDataset(
        train_data.tolist(),
        train_target.tolist(),
        tokenizer=tokenizer,
        label_to_id=LABEL_TO_ID,
        max_len=128,
    )

    test_dataset = src.datasets.SentimentAnalysisExcelDataset(
        test_data.tolist(),
        test_target.tolist(),
        tokenizer=tokenizer,
        label_to_id=LABEL_TO_ID,
        max_len=128,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )
    valid_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    logger.info("Initializing BertForSequenceClassification from pretrained BERT LM")
    model = BertForSequenceClassification.from_pretrained(
        args.path_to_bert,
        num_labels=len(train_dataset.label_to_id),
        output_attentions=False,
        output_hidden_states=False,
    )

    if args.full_finetuning:
        param_optimizer = list(model.named_parameters())
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
        param_optimizer = list(model.classifier.named_parameters())
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
        main_metric=src.metrics.F1Score(len(train_dataset.label_to_id)),
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
            src.metrics.RecallScore(len(train_dataset.label_to_id)),
            src.metrics.PrecisionScore(len(train_dataset.label_to_id)),
        ],
    )
    trainer.fit()
