from dataclasses import dataclass
import itertools
import typing as T

import numpy as np
import pandas as pd

from spooky_trees.criterions import SpookyCriterion
from spooky_trees.predicates import (
    SpookyCategorialPredicate,
    SpookyCompareOperation,
    SpookyNumericalPredicate,
    SpookyPredicate,
)


@dataclass
class SpookyNode:
    children: T.Tuple[T.Tuple[SpookyPredicate, "SpookyNode"]] = ()

    def __init__(self, prediction: np.ndarray):
        self._prediction = prediction
    
    def predict_proba(self) -> np.ndarray:
        return self._prediction
    
    def predict(self) -> int:
        return self.predict_proba().argmax()


class SpookyTree:
    def __init__(
        self,
        criterion: T.Type[SpookyCriterion],
        max_depth: int = -1,
        min_samples_split: int = 2,
        min_information_gain: float = 0.0,
        rsm: float = 1.0,
        cat_features: T.Optional[T.Set[int]] = None,
        **criterion_params,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.rsm = rsm
        self.cat_features = cat_features or set()
        self.criterion = criterion(**criterion_params)

        self._root = None
    
    def _get_best_numerical_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        feature_idx: str
    ) -> T.Tuple[T.Union[int, float], float]:
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
            if idx >= len(X_feature_sorted):
                break
            
            left_weights = weights[:idx]
            right_weights = weights[idx:]

            y_left_predictions = self.criterion.predict(y_sorted[:idx], weights=left_weights)
            y_right_predictions = self.criterion.predict(y_sorted[idx:], weights=right_weights)

            left_criterion = self.criterion(y_sorted[:idx], y_left_predictions, weights=left_weights)
            right_criterion = self.criterion(y_sorted[idx:], y_right_predictions, weights=right_weights)

            criterion = left_criterion * idx + right_criterion * (len(X_feature_sorted) - idx)
            if criterion < best_criterion:
                best_criterion = criterion
                best_threshold = X_feature_sorted[idx]
            idx += 1

        return best_threshold, best_criterion
    
    def _get_powerset(self, iterable: T.Iterable) -> T.List[T.Iterable]:
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
    
    def _get_best_categorial_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray, 
        feature_idx: int
    ) -> T.Tuple[T.List[T.Set[float]], float]:
        unique_values = set(np.unique(X[:, feature_idx].astype(int)))
        if len(unique_values) == 1:
            return None, 1e9

        best_criterion = 0
        threshold = [{value} for value in unique_values]
        for feature_value in unique_values:
            feature_value_mask = X[:, feature_idx] == feature_value
            y_feature_value = y[feature_value_mask]
            weights_feature_value = weights[feature_value_mask]

            y_feature_value_predictions = self.criterion.predict(y_feature_value, weights=weights_feature_value)
            best_criterion += self.criterion(y_feature_value, y_feature_value_predictions, weights=weights_feature_value) * len(y_feature_value)

        # for feature_value in unique_values:
        #     feature_value_mask = X[:, feature_idx] == feature_value

        #     y_feature_value = y[feature_value_mask]
        #     y_not_feature_value = y[~feature_value_mask]

        #     y_feature_value_predictions = self.criterion.predict(y_feature_value)
        #     y_not_feature_value_predictions = self.criterion.predict(y_not_feature_value)
        #     current_criterion = self.criterion(y_feature_value, y_feature_value_predictions) * len(y_feature_value)
        #     current_criterion += self.criterion(y_not_feature_value, y_not_feature_value_predictions) * len(y_not_feature_value)

        #     if current_criterion < best_criterion:
        #         best_criterion = current_criterion
        #         threshold = [{feature_value}, unique_values.difference((feature_value,))]
        
        # for features_set in self._get_powerset(unique_values):
        #     if len(features_set) == 0 or len(features_set) == len(unique_values):
        #         continue
        #     features_set = set(features_set)
        #     vfunc = np.vectorize(lambda x: x in features_set)

        #     feature_value_mask = vfunc(X[:, feature_idx])

        #     y_feature_value = y[feature_value_mask]
        #     y_not_feature_value = y[~feature_value_mask]

        #     y_feature_value_predictions = self.criterion.predict(y_feature_value)
        #     y_not_feature_value_predictions = self.criterion.predict(y_not_feature_value)
        #     current_criterion = self.criterion(y_feature_value, y_feature_value_predictions) * len(y_feature_value)
        #     current_criterion += self.criterion(y_not_feature_value, y_not_feature_value_predictions) * len(y_not_feature_value)

        #     if current_criterion < best_criterion:
        #         best_criterion = current_criterion
        #         threshold = [features_set, unique_values.difference(features_set)]

        return threshold, best_criterion

    def _get_best_feature(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> T.Tuple[str, T.Optional[float]]:
        best_feature = None
        best_threshold = None
        best_criterion = None

        allowed_features = np.random.binomial(1, self.rsm, size=X.shape[1])
        for idx in range(X.shape[1]):
            if allowed_features[idx] == 0:
                continue

            if idx in self.cat_features:
                threshold, current_criterion = self._get_best_categorial_threshold(X, y, weights, idx)
            else:
                threshold, current_criterion = self._get_best_numerical_threshold(X, y, weights, idx)

            if best_criterion is None or current_criterion < best_criterion:
                best_feature = idx
                best_threshold = threshold
                best_criterion = current_criterion
        
        return best_feature, best_threshold, best_criterion

    def _spooky_branch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        node: T.Optional[SpookyNode],
        depth: int
    ):
        assert len(X) == len(y), "Lengths of samples and targets should be equal"

        if self.max_depth > 0 and depth >= self.max_depth:
            return
        if len(np.unique(y)) == 1:
            return
        if len(X) < self.min_samples_split:
            return

        feature_idx, threshold, criterion = self._get_best_feature(X, y, weights)

        y_predictions = self.criterion.predict(y, weights=weights)
        if len(y) * self.criterion(y, y_predictions, weights=weights) - criterion < self.min_information_gain:
            return

        children = []
        if feature_idx in self.cat_features:
            for features_set in threshold:
                feature_value_predicate = SpookyCategorialPredicate(feature_idx, features_set)
                feature_value_mask = feature_value_predicate(X)

                prediction = self.criterion.predict(y[feature_value_mask], weights=weights[feature_value_mask])
                feature_value_node = SpookyNode(prediction=prediction)

                children.append((feature_value_predicate, feature_value_node))

                self._spooky_branch(
                    X=X[feature_value_mask],
                    y=y[feature_value_mask],
                    weights=weights[feature_value_mask],
                    node=feature_value_node,
                    depth=depth + 1,
                )
        else:
            assert threshold is not None
            left_predicate = SpookyNumericalPredicate(feature_idx, threshold, SpookyCompareOperation.LESS)
            right_predicate = SpookyNumericalPredicate(feature_idx, threshold, SpookyCompareOperation.GREATER_OR_EQUAL)

            left_elements_mask = left_predicate(X)
            right_elements_mask = right_predicate(X)

            left_prediction = self.criterion.predict(y[left_elements_mask], weights=weights[left_elements_mask])
            right_prediction = self.criterion.predict(y[right_elements_mask], weights=weights[right_elements_mask])

            left_node = SpookyNode(prediction=left_prediction)
            right_node = SpookyNode(prediction=right_prediction)

            children.append((left_predicate, left_node))
            children.append((right_predicate, right_node))

            self._spooky_branch(
                X=X[left_elements_mask],
                y=y[left_elements_mask],
                weights=weights[left_elements_mask],
                node=left_node,
                depth=depth + 1,
            )
            self._spooky_branch(
                X=X[right_elements_mask],
                y=y[right_elements_mask],
                weights=weights[right_elements_mask],
                node=right_node,
                depth=depth + 1,
            )

        node.children = tuple(children)

    def fit(self, X: np.ndarray, y: np.ndarray, weights: T.Optional[np.ndarray] = None) -> "SpookyTree":
        assert len(X) == len(y), "Number of objects and targets should be equal"
        
        if weights is None:
            weights = np.ones(len(y), dtype=np.float32)

        self.n_classes = len(np.unique(y))

        prediction = self.criterion.predict(y, weights)
        self._root = SpookyNode(prediction=prediction)
        self._spooky_branch(X=X, y=y, weights=weights, node=self._root, depth=1)

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
