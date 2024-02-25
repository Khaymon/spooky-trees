from dataclasses import dataclass

import pandas as pd

from .spooky_predicate import SpookyPredicate


@dataclass
class SpookyCompareOperation:
    LESS = "less_than"
    GREATER_OR_EQUAL = "greater_or_equal"


class SpookyNumericalPredicate(SpookyPredicate):
    def __init__(self, feature: str, value: float | int, operation: str):
        assert operation in (SpookyCompareOperation.LESS, SpookyCompareOperation.GREATER_OR_EQUAL)
        
        self._feature = feature
        self._value = value
        self._operation = operation

    def __call__(self, X: pd.DataFrame | pd.Series) -> bool:
        if self._operation == SpookyCompareOperation.LESS:
            return X[self._feature] < self._value
        elif self._operation == SpookyCompareOperation.GREATER_OR_EQUAL:
            return X[self._feature] >= self._value
        else:
            raise NotImplementedError(f"Operation {self._operation} is not implemented")
