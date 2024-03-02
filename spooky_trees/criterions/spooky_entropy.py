import numpy as np
import pandas as pd
import typing as T

from .spooky_criterion import SpookyCriterion


class SpookyEntropy(SpookyCriterion):
    def __init__(self, n_classes: int, **kwargs):
        super().__init__(**kwargs)

        self._n_classes = n_classes

    def probas(self, classes: np.ndarray) -> np.ndarray:
        if len(classes) == 0:
            return np.zeros(self._n_classes, dtype=np.float32)

        if classes.ndim == 2:
            counts = classes.sum(axis=0)
            assert len(counts) == self._n_classes
            probas = counts / counts.sum()
        else:
            classes, counts = np.unique(classes, return_counts=True)
            probas = np.zeros(self._n_classes, dtype=np.float32)
            probas[classes] = counts / counts.sum()

        return probas

    def __call__(self, y: np.ndarray, predictions: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0

        non_zero_mask = predictions != 0
        return np.mean(-self.probas(y)[non_zero_mask] * np.log(predictions[non_zero_mask]))

    def predict(self, y: np.ndarray) -> float:
        return self.probas(y)

    def grad_output(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        return -y / (predictions + 1e-5)
