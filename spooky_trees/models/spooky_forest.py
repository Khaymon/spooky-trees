from functools import partial
from multiprocessing import Pool
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
        n_workers: int | None = None,
        **kwargs
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_workers = n_workers

        self.spooky_tree_params = kwargs

        self._spooky_trees: T.List[SpookyTree] = []

    @staticmethod
    def _train_spooky_tree(ids: T.Sequence[int], X: np.ndarray, y: np.ndarray, tree_params: T.Dict) -> SpookyTree:
        return SpookyTree(**tree_params).fit(X[ids], y[ids])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SpookyForest":
        self.max_samples = self.max_samples or len(X)
        
        with Pool(self.n_workers) as pool:
            ids = [np.random.choice(np.arange(len(X)), size=self.max_samples) for _ in range(self.n_estimators)]
            self._spooky_trees = list(tqdm(pool.imap_unordered(
                partial(self._train_spooky_tree, X=X, y=y, tree_params=self.spooky_tree_params), ids
            ), total=len(ids)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = []
        for spooky_tree in tqdm(self._spooky_trees):
            probas.append(spooky_tree.predict_proba(X))
        
        return np.mean(np.stack(probas), axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=-1)
