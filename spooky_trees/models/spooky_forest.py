import numpy as np
import pandas as pd
from tqdm import tqdm
import typing as T

from .spooky_tree import SpookyTree


class SpookyForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int | None = None,
        **kwargs
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples

        self.spooky_tree_params = kwargs

        self._spooky_trees: T.List[SpookyTree] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SpookyForest":
        self.max_samples = self.max_samples or len(X)
        self._spooky_trees = []
        for _ in tqdm(range(self.n_estimators)):
            spooky_tree = SpookyTree(**self.spooky_tree_params)
            current_ids = np.random.choice(np.arange(len(X)), size=self.max_samples)
            self._spooky_trees.append(spooky_tree.fit(X.iloc[current_ids], y.iloc[current_ids]))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = []
        for spooky_tree in tqdm(self._spooky_trees):
            probas.append(spooky_tree.predict_proba(X))
        
        return np.mean(np.stack(probas), axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=-1)
