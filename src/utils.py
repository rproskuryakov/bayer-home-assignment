import random
from typing import List
from typing import Union

import numpy as np
import pandas as pd
import torch
import transformers.file_utils
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


def pad_sequence(sequence, max_len=75, value: Union[str, int] = "SEP"):
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + (max_len - len(sequence)) * [value]


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if transformers.file_utils.is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if transformers.file_utils.is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)


def tokenize_sentence_pair(
    first_sentence: str,
    second_sentence: str,
    tokenizer: BertTokenizer,
) -> (List[int], List[int]):
    first_sentence_ids = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(first_sentence)
    )
    second_sentence_ids = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(second_sentence)
    )
    encoded_pair = tokenizer.build_inputs_with_special_tokens(
        first_sentence_ids,
        second_sentence_ids,
    )
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(
        first_sentence_ids,
        second_sentence_ids,
    )
    return encoded_pair, token_type_ids


def predict_label_for_aux_sentence_model(text, model, tokenizer, label_to_statement):
    text_probabilities = {}
    for key in label_to_statement:
        encoded_pair, token_type_ids = tokenize_sentence_pair(
            text,
            label_to_statement[key],
            tokenizer=tokenizer,
        )
        attention_mask = [float(i != tokenizer.pad_token_id) for i in encoded_pair]
        with torch.no_grad():
            outputs = model(
                torch.LongTensor(encoded_pair).reshape(1, -1),
                token_type_ids=torch.LongTensor(token_type_ids).reshape(-1, 1),
                attention_mask=torch.FloatTensor(attention_mask).reshape(-1, 1),
                return_dict=True,
            )
        text_probabilities[key] = outputs.logits[0, 0].detach().cpu().numpy()

    best_cat, _ = max(text_probabilities.items(), key=lambda x: x[1])
    return best_cat


def get_train_test_split(data_path, random_seed: int = 0):
    df = pd.read_excel(data_path, index_col="ID").rename(
        columns={
            "Sentence": "text",
            "Positive": "positive",
            "Negative": "negative",
            "Neutral": "neutral",
        }
    )
    df["target"] = df.apply(
        lambda x: "positive"
        if x.positive
        else ("negative" if x.negative else "neutral"),
        axis=1,
    )

    # Split data on train and test set
    return train_test_split(
        df.text, df.target, shuffle=True, random_state=random_seed, stratify=df.target
    )
