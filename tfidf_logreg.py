import argparse
import logging
import pathlib
import sys

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import src.utils

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train simple Logistic Regression classifier on top of Tf-Idf features."
    )
    parser.add_argument("dataset_path", type=pathlib.Path)
    args = parser.parse_args()

    RANDOM_SEED = 6042022
    src.utils.set_seed(RANDOM_SEED)

    train_data, test_data, train_target, test_target = src.utils.get_train_test_split(
        args.dataset_path, RANDOM_SEED
    )
    logger.info("Dataset loaded.")

    target_encoder = LabelEncoder()
    train_encoded_labels = target_encoder.fit_transform(train_target)
    test_encoded_labels = target_encoder.transform(test_target)

    logger.info("Starting to train model...")
    estimator = Pipeline([("tfidf", TfidfVectorizer(stop_words=stopwords.words("english"))),
                          ("clf", LogisticRegression(multi_class="multinomial"))])
    estimator.fit(train_data, train_encoded_labels)
    logger.info("Process of model training finished.")

    test_predictions = estimator.predict(test_data)

    logger.info(
        "Macro Recall: %.2f",
        recall_score(test_encoded_labels, test_predictions, average="macro")
    )
    logger.info(
        "Macro Precision: %.2f",
        precision_score(test_encoded_labels, test_predictions, average="macro")
    )
    logger.info(
        "Macro F1: %.2f",
        f1_score(test_encoded_labels, test_predictions, average="macro")
    )
