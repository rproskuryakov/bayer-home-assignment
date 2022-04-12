from typing import Dict
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertTokenizer
import torch

from src.utils import pad_sequence

# tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
#
# model = AutoModelForMaskedLM.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
from src.utils import tokenize_sentence_pair


class SentimentAnalysisExcelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer: BertTokenizer,
        max_len: int = 75,
        label_to_id: Dict[str, int] = None,
    ):
        self.sequences = texts
        self.labels = labels
        self.label_to_id = label_to_id
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        sentence, label = self.sequences[item], self.labels[item]
        tokens = self.tokenizer.tokenize(f"[CLS] {sentence} [SEP]")
        input_ids = pad_sequence(
            self.tokenizer.convert_tokens_to_ids(tokens),
            max_len=self.max_len,
            value=self.tokenizer.pad_token_id,
        )
        attention_mask = [float(i != 0.0) for i in input_ids]
        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_masks": torch.FloatTensor(attention_mask),
            "labels": self.label_to_id[label],
        }

    def __len__(self):
        return len(self.sequences)


class SentimentAnalysisPairClassificationTask(torch.utils.data.Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer: BertTokenizer,
        max_len: int = 256,
        label_to_statement: Dict[str, str] = None,
    ):
        self.items = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_to_statement = label_to_statement
        for text, label in zip(texts, labels):
            for key in label_to_statement:
                self.items.append((text, label_to_statement[key]))
                if label == key:
                    self.labels.append(0)
                else:
                    self.labels.append(1)

    def __getitem__(self, item):
        encoded_pair, token_type_ids = tokenize_sentence_pair(
            *self.items[item], tokenizer=self.tokenizer
        )
        padded = pad_sequence(
            encoded_pair, max_len=self.max_len, value=self.tokenizer.pad_token_id
        )
        attention_mask = [float(i != self.tokenizer.pad_token_id) for i in padded]
        padded_token_type_ids = pad_sequence(
            token_type_ids,
            max_len=self.max_len,
            value=1,
        )
        return {
            "input_ids": torch.LongTensor(padded),
            "attention_masks": torch.FloatTensor(attention_mask),
            "token_type_ids": torch.LongTensor(padded_token_type_ids),
            "labels": self.labels[item],
        }

    def __len__(self):
        return len(self.items)
