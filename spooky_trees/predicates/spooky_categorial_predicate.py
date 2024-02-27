from dataclasses import dataclass
import typing as T

import numpy as np

from .spooky_predicate import SpookyPredicate


class SpookyCategorialPredicate(SpookyPredicate):
    def __init__(self, feature_idx: int, feature_values: T.Set[float]):
        self.feature_idx = feature_idx

        self.predicate_func = np.vectorize(lambda x: x in feature_values)

    def __call__(self, X: np.ndarray) -> np.ndarray | bool:

        if X.ndim == 2:
            return self.predicate_func(X[:, self.feature_idx].astype(int))
        elif X.ndim == 1:
            return self.predicate_func(int(X[self.feature_idx]))
        else:
            raise NotImplementedError()
