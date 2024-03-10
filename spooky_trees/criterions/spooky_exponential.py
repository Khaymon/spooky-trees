import numpy as np
import pandas as pd
import typing as T

from .spooky_criterion import SpookyCriterion


class SpookyExponential(SpookyCriterion):
    def __init__(self, n_classes: int, **kwargs):
        super().__init__(**kwargs)

        self._n_classes = n_classes

    def probas(self, classes: np.ndarray, weights: T.Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if weights is None:
            weights = np.ones(len(classes), dtype=np.float32)
        if len(classes) == 0:
            return np.zeros(self._n_classes, dtype=np.float32)

        if classes.ndim == 2:
            counts = (weights[:, None] * classes).sum(axis=0) 
            assert len(counts) == self._n_classes
            probas = counts / counts.sum()
        else:
            raise NotImplementedError()

        return probas
    
    def _exp(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        non_zero_mask = predictions != 0
        
        try:
            return np.exp(np.sum(-self.probas(y)[non_zero_mask] * predictions[non_zero_mask]))
        except:
            print(predictions.shape, self.probas(y).shape)
            assert False

    def __call__(self, y: np.ndarray, predictions: np.ndarray, weights: T.Optional[np.ndarray] = None, **kwargs) -> float:
        if len(y) == 0:
            return 0.0
        if weights is None:
            weights = np.ones_like(y) / len(y)

        return np.sum(weights * self._exp(y, predictions))

    def predict(self, y: np.ndarray, weights: T.Optional[np.ndarray] = None, **kwargs) -> float:
        return self.probas(y, weights)

    def grad_output(self, y: np.ndarray, predictions: np.ndarray, weights: T.Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if weights is None:
            weights = np.ones_like(y) / len(y)

        return -y * weights * self._exp(y, predictions)
