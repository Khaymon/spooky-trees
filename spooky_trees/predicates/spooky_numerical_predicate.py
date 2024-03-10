from dataclasses import dataclass
import typing as T

import numpy as np

from .spooky_predicate import SpookyPredicate


@dataclass
class SpookyCompareOperation:
    LESS = "less_than"
    GREATER_OR_EQUAL = "greater_or_equal"


class SpookyNumericalPredicate(SpookyPredicate):
    def __init__(self, feature_idx: str, value: T.Union[float, int], operation: str):
        assert operation in (SpookyCompareOperation.LESS, SpookyCompareOperation.GREATER_OR_EQUAL)
        
        self._feature_idx = feature_idx
        self._value = value
        self._operation = operation

    def __call__(self, X: np.ndarray) -> T.Union[np.ndarray, bool]:
        if self._operation == SpookyCompareOperation.LESS:
            if X.ndim == 2:
                return X[:, self._feature_idx] < self._value
            elif X.ndim == 1:
                return X[self._feature_idx] < self._value
            else:
                raise RuntimeError()
        elif self._operation == SpookyCompareOperation.GREATER_OR_EQUAL:
            if X.ndim == 2:
                return X[:, self._feature_idx] >= self._value
            elif X.ndim == 1:
                return X[self._feature_idx] >= self._value
            else:
                raise RuntimeError()
        else:
            raise NotImplementedError(f"Operation {self._operation} is not implemented")
