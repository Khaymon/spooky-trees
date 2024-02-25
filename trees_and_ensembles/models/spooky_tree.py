from dataclasses import dataclass
import typing as T

import numpy as np
import pandas as pd

from trees_and_ensembles.criterions import SpookyCriterion, SpookyEntropy
from trees_and_ensembles.predicates import SpookyCategorialPredicate, SpookyPredicate


@dataclass
class SpookyNode:
    children: T.Tuple[T.Tuple[SpookyPredicate, "SpookyNode"]] = ()

    def __init__(self, y: pd.Series):
        y_counts = y.value_counts()

        self.prediction = y_counts.index[y_counts.argmax()]

    def predict(self) -> T.Any:
        return self.prediction


class SpookyTree:
    def __init__(self, criterion: SpookyCriterion = SpookyEntropy(), max_depth: int = -1):
        self.max_depth = max_depth
        self.criterion = criterion

        self._root = None
    
    def _get_best_numerical_threshold(self, X: pd.DataFrame, y: pd.Series, feature: str) -> T.Tuple[int | float, float]:
        argsorted_feature_values = X[feature].to_numpy().argsort()

        sorted_feature_values = X[feature].iloc[argsorted_feature_values]
        y_sorted = y.iloc[argsorted_feature_values]

        best_threshold = None
        best_criterion = None

        # Naiive implementation with O(n^2) complexity
        for idx in range(1, len(sorted_feature_values)):
            left_y = y_sorted[:idx]
            right_y = y_sorted[idx:]

            left_criterion = self.criterion(left_y)
            right_criterion = self.criterion(right_y)
            if len(left_y) == 0:
                current_criterion = right_criterion
            elif len(right_criterion) == 0:
                current_criterion = left_criterion
            else:
                current_criterion = left_criterion * len(left_y) + right_criterion * len(right_y)
            
            if best_criterion is None or current_criterion < best_criterion:
                best_threshold = sorted_feature_values[idx]
                best_criterion = current_criterion
        
        return best_threshold, best_criterion

    def _get_best_feature(self, X: pd.DataFrame, y: pd.Series) -> T.Tuple[str, float | None]:
        best_feature = None
        best_threshold = None
        best_criterion = None
        for feature in X:
            current_criterion = 0
            if X[feature].dtype == object:
                for feature_value in X[feature].unique():
                    feature_value_mask = X[feature] == feature_value
                    y_feature_value = y[feature_value_mask]

                    current_criterion += self.criterion(y_feature_value) * len(y_feature_value)
                threshold = None  # None for categorial features
            elif X[feature].dtype in (int, float):
                threshold, current_criterion = self._get_best_numerical_threshold(X, y, feature)
            else:
                raise NotImplementedError()

            if best_criterion is None or current_criterion < best_criterion:
                best_feature = feature
                best_threshold = threshold
                best_criterion = current_criterion
        
        return best_feature, best_threshold

    def _spooky_branch(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        node: SpookyNode | None,
        depth: int
    ):
        assert len(X) == len(y), "Lengths of samples and targets should be equal"

        if self.max_depth > 0 and depth >= self.max_depth:
            return
        if len(y.unique()) == 1:
            return

        feature, threshold = self._get_best_feature(X, y)

        children = []
        if X[feature].dtype == object:
            for feature_value in X[feature].unique():
                feature_value_predicate = SpookyCategorialPredicate(feature, feature_value)
                feature_value_mask = feature_value_predicate(X)
                feature_value_node = SpookyNode(y[feature_value_mask])

                children.append((feature_value_predicate, feature_value_node))

                self._spooky_branch(
                    X=X[feature_value_mask],
                    y=y[feature_value_mask],
                    node=feature_value_node,
                    depth=depth + 1,
                )
        elif X[feature].dtype in (float, int):
            left_predicate = SpookyCategorialPredicate(feature, threshold)
            right_predicate = SpookyCategorialPredicate(feature, threshold)

            left_node = SpookyNode(y[left_elements_mask])
            right_node = SpookyNode(y[right_elements_mask])

            children.append((left_predicate, left_node))
            children.append((right_predicate, right_node))

            left_elements_mask = left_predicate(X)
            right_elements_mask = right_predicate(X)

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
        else:
            raise NotImplementedError()

        node.children = tuple(children)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SpookyTree":
        assert len(X) == len(y), "Number of objects and targets should be equal"

        self._root = SpookyNode(y)
        self._spooky_branch(X=X, y=y, node=self._root, depth=1)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = [None] * len(X)

        for idx in range(len(X)):
            current_node = self._root
            while len(current_node.children) > 0:
                node_changed = False
                for predicate, child in current_node.children:
                    if predicate(X.iloc[idx]):
                        current_node = child
                        node_changed = True

                        break
                if not node_changed:
                    raise RuntimeError("Node is not changed during traversal")
            predictions[idx] = current_node.predict()
        
        return predictions
