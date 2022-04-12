from abc import ABC
from abc import abstractmethod

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class BaseMetric(ABC):
    name: str

    @abstractmethod
    def __call__(self, ground_labels, pred_labels, *args, **kwargs):
        pass

    @abstractmethod
    def calculate(self):
        pass


class PrecisionScore(BaseMetric):
    name = "PrecisionScore"

    def __init__(self, n_labels: int = 2):
        self.ground_truth = []
        self.predicted_labels = []
        self.average = "binary" if n_labels == 2 else "macro"

    def __call__(self, ground_labels, pred_labels, *args, **kwargs):
        for item in ground_labels:
            self.ground_truth.append(item)
        for item in pred_labels:
            self.predicted_labels.append(item)

    def calculate(self):
        metric = precision_score(
            self.ground_truth, self.predicted_labels, average=self.average
        )
        self.ground_truth, self.predicted_labels = [], []
        return metric


class RecallScore(BaseMetric):
    name = "RecallScore"

    def __init__(self, n_labels: int = 2):
        self.ground_truth = []
        self.predicted_labels = []
        self.average = "binary" if n_labels == 2 else "macro"

    def __call__(self, ground_labels, pred_labels, *args, **kwargs):
        for item in ground_labels:
            self.ground_truth.append(item)
        for item in pred_labels:
            self.predicted_labels.append(item)

    def calculate(self):
        metric = recall_score(
            self.ground_truth, self.predicted_labels, average=self.average
        )
        self.ground_truth, self.predicted_labels = [], []
        return metric


class F1Score(BaseMetric):
    name = "F1Score"

    def __init__(self, n_labels: int = 2):
        self.ground_truth = []
        self.predicted_labels = []
        self.average = "binary" if n_labels == 2 else "macro"

    def __call__(self, ground_labels, pred_labels, *args, **kwargs):
        for item in ground_labels:
            self.ground_truth.append(item)
        for item in pred_labels:
            self.predicted_labels.append(item)

    def calculate(self):
        metric = f1_score(
            self.ground_truth, self.predicted_labels, average=self.average
        )
        self.ground_truth, self.predicted_labels = [], []
        return metric
