import numpy as np
import pandas as pd
import typing as T

from .preprocessor import Preprocessor


class LabelEncoder(Preprocessor):
    def __init__(self, cat_features: T.List[str]):
        self._cat_features = cat_features
        self._mappings = {}

    def fit(self, X: pd.DataFrame):
        for feature in self._cat_features:
            self._mappings[feature] = {value: idx for idx, value in enumerate(X[feature].unique())} 
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_result = X.copy()
        for feature in self._cat_features:
            X_result[feature] = X_result[feature].map(self._mappings[feature])

        return X_result.to_numpy()

