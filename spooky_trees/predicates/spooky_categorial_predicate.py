from dataclasses import dataclass

import numpy as np

from .spooky_predicate import SpookyPredicate


@dataclass
class SpookyCategorialPredicate(SpookyPredicate):
    feature_idx: int
    feature_value: float

    def __call__(self, X: np.ndarray) -> np.ndarray | bool:
        if X.ndim == 2:
            return X[:, self.feature_idx] == self.feature_value
        elif X.ndim == 1:
            return X[self.feature_idx] == self.feature_value
        else:
            raise NotImplementedError()
