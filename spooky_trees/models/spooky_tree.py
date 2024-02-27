from dataclasses import dataclass
import typing as T

import numpy as np
import pandas as pd

from spooky_trees.preprocess import Preprocessor, LabelEncoder
from spooky_trees.criterions import SpookyCriterion, SpookyEntropy
from spooky_trees.predicates import (
    SpookyCategorialPredicate,
    SpookyCompareOperation,
    SpookyNumericalPredicate,
    SpookyPredicate,
)


@dataclass
class SpookyNode:
    children: T.Tuple[T.Tuple[SpookyPredicate, "SpookyNode"]] = ()

    def __init__(self, n_classes: int, y: np.ndarray):
        assert len(y) >= 1, "Targets length should be greater or equal to 1"

        self.n_classes = n_classes
        values, counts = np.unique(y, return_counts=True)

        self._values = values
        self._proba_estimation = counts / len(y)
    
    def predict_proba(self) -> np.ndarray:
        probas = np.zeros(self.n_classes, dtype=float)
        probas[self._values] = self._proba_estimation

        return probas
    
    def predict(self) -> int:
        return self.predict_proba().argmax()


class SpookyTree:
    def __init__(
        self,
        criterion: SpookyCriterion = SpookyEntropy(),
        max_depth: int = -1,
        min_samples_split: int = 2,
        min_information_gain: float = 0.0,
        cat_features: T.Set[int] | None = None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.cat_features = cat_features or set()

        self._root = None
    
    def _get_best_numerical_threshold(self, X: np.ndarray, y: np.ndarray, feature_idx: str) -> T.Tuple[int | float, float]:
        assert len(y) > 1, "Number of samples must be greater than 1"

        argsorted_feature_values = X[:, feature_idx].argsort()

        X_feature_sorted = X[argsorted_feature_values, feature_idx]
        y_sorted = y[argsorted_feature_values]

        best_threshold = None
        best_criterion = 1e9

        idx = 1
        while idx < len(X_feature_sorted) and X_feature_sorted[idx - 1] == X_feature_sorted[idx]:
            idx += 1

        while idx < len(X_feature_sorted):
            while idx < len(X_feature_sorted) and X_feature_sorted[idx] == X_feature_sorted[idx - 1]:
                idx += 1

            left_criterion = self.criterion(y_sorted[:idx])
            right_criterion = self.criterion(y_sorted[idx:])

            criterion = left_criterion * idx + right_criterion * (len(X_feature_sorted) - idx)
            if criterion < best_criterion:
                best_criterion = criterion
                best_threshold = X_feature_sorted[idx]
            idx += 1

        return best_threshold, best_criterion
    
    def _get_best_categorial_threshold(self, X: np.ndarray, y: np.ndarray, feature_idx: int) -> T.Tuple[None, float]:
        current_criterion = 0
        for feature_value in np.unique(X[:, feature_idx]):
            feature_value_mask = X[:, feature_idx] == feature_value
            y_feature_value = y[feature_value_mask]

            current_criterion += self.criterion(y_feature_value) * len(y_feature_value)

        return None, current_criterion

    def _get_best_feature(self, X: np.ndarray, y: np.ndarray) -> T.Tuple[str, float | None]:
        best_feature = None
        best_threshold = None
        best_criterion = None
        for idx in range(X.shape[1]):
            if idx in self.cat_features:
                threshold, current_criterion = self._get_best_categorial_threshold(X, y, idx)
            else:
                threshold, current_criterion = self._get_best_numerical_threshold(X, y, idx)

            if best_criterion is None or current_criterion < best_criterion:
                best_feature = idx
                best_threshold = threshold
                best_criterion = current_criterion
        
        return best_feature, best_threshold, best_criterion

    def _spooky_branch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        node: SpookyNode | None,
        depth: int
    ):
        assert len(X) == len(y), "Lengths of samples and targets should be equal"

        if self.max_depth > 0 and depth >= self.max_depth:
            return
        if len(np.unique(y)) == 1:
            return
        if len(X) < self.min_samples_split:
            return

        feature_idx, threshold, criterion = self._get_best_feature(X, y)

        if len(y) * self.criterion(y) - criterion < self.min_information_gain:
            return

        children = []
        if feature_idx in self.cat_features:
            for feature_value in np.unique(X[:, feature_idx]):
                feature_value_predicate = SpookyCategorialPredicate(feature_idx, feature_value)
                feature_value_mask = feature_value_predicate(X)
                feature_value_node = SpookyNode(n_classes=self.n_classes, y=y[feature_value_mask])

                children.append((feature_value_predicate, feature_value_node))

                self._spooky_branch(
                    X=X[feature_value_mask],
                    y=y[feature_value_mask],
                    node=feature_value_node,
                    depth=depth + 1,
                )
        else:
            assert threshold is not None
            left_predicate = SpookyNumericalPredicate(feature_idx, threshold, SpookyCompareOperation.LESS)
            right_predicate = SpookyNumericalPredicate(feature_idx, threshold, SpookyCompareOperation.GREATER_OR_EQUAL)

            left_elements_mask = left_predicate(X)
            right_elements_mask = right_predicate(X)

            left_node = SpookyNode(n_classes=self.n_classes, y=y[left_elements_mask])
            right_node = SpookyNode(n_classes=self.n_classes, y=y[right_elements_mask])

            children.append((left_predicate, left_node))
            children.append((right_predicate, right_node))

            self._spooky_branch(
                X=X[left_elements_mask],
                y=y[left_elements_mask],
                node=left_node,
                depth=depth + 1,
            )
            self._spooky_branch(
                X=X[right_elements_mask],
                y=y[right_elements_mask],
                node=right_node,
                depth=depth + 1,
            )

        node.children = tuple(children)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpookyTree":
        assert len(X) == len(y), "Number of objects and targets should be equal"

        self.n_classes = len(np.unique(y))

        self._root = SpookyNode(n_classes=self.n_classes, y=y)
        self._spooky_branch(X=X, y=y, node=self._root, depth=1)

        return self
    
    def _find_node(self, x: pd.Series) -> SpookyNode:
        current_node = self._root
        while len(current_node.children) > 0:
            node_changed = False
            for predicate, child in current_node.children:
                if predicate(x):
                    current_node = child
                    node_changed = True
                    break
            if not node_changed:
                break

        return current_node

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = []
        for idx in range(len(X)):
            node = self._find_node(X[idx])
            probas.append(node.predict_proba())

        return np.vstack(probas)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=-1)
