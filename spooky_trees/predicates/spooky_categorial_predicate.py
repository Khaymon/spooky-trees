from dataclasses import dataclass

import pandas as pd

from .spooky_predicate import SpookyPredicate


@dataclass
class SpookyCategorialPredicate(SpookyPredicate):
    feature_name: str
    feature_value: str

    def __call__(self, X: pd.DataFrame | pd.Series) -> bool | pd.Series:
        return X[self.feature_name] == self.feature_value
