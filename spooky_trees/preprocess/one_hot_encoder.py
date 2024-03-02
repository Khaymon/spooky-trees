import typing as T

import numpy as np
import pandas as pd

from .preprocessor import Preprocessor


class OneHotEncoder(Preprocessor):
    def __init__(self, transform_columns: T.Sequence[str]):
        self._transform_columns = transform_columns

    def fit(self, X: pd.DataFrame):
        self._all_columns = X.columns

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return pd.get_dummies(X[self._all_columns], columns=self._transform_columns).to_numpy().astype(float)
