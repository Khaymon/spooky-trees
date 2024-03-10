from copy import deepcopy
from tqdm import tqdm
import typing as T

import numpy as np

from spooky_trees.criterions import SpookyCriterion, SpookyExponential
from spooky_trees.metrics import cross_entropy, f1_score
from spooky_trees.model_selection import train_test_split
from .spooky_tree import SpookyTree


class AdaSpookyBoost:
    def __init__(
        self,
        n_estimators: int,
        learning_rate: float = 1e-1,
        val_size: float = 0.05,
        early_stop: bool = False,
        early_stop_patience: int = -1,
        val_metric: T.Callable[[np.ndarray, np.ndarray], float] = f1_score,
        **kwargs
    ):
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._val_size = val_size

        if early_stop:
            assert early_stop_patience >= 0
            self._early_stop_patience = early_stop_patience
        else:
            self._early_stop_patience = 1e9

        self._val_metric = val_metric

        self.spooky_tree_params = kwargs

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        return np.exp(logits) / (np.sum(np.exp(logits), axis=1, keepdims=True))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaSpookyBoost":
        assert y.ndim == 2

        X_train, X_val, y_train, y_val = train_test_split(X, y, self._val_size)

        self._estimators = []

        train_prediction = np.zeros_like(y_train)
        val_prediction = np.zeros_like(y_val)

        max_val_metric = -1e9
        last_min_val_metric_update = 0
        
        tree_params = deepcopy(self.spooky_tree_params)
        tree_params["criterion"] = SpookyExponential
        
        current_weights = np.ones(len(y_train), dtype=np.float32)

        progress_bar = tqdm(range(self._n_estimators))
        try:
            for idx in progress_bar:
                current_estimator = SpookyTree(**tree_params).fit(X_train, y_train, weights=current_weights)
                self._estimators.append(current_estimator)

                current_train_prediction = self._learning_rate * current_estimator.predict_proba(X_train)
                current_val_prediction = self._learning_rate * current_estimator.predict_proba(X_val)
                
                train_prediction += current_train_prediction
                val_prediction += current_val_prediction

                val_probas = self._softmax(val_prediction)
                val_metric = self._val_metric(y_val, val_probas)
                progress_bar.set_description("val metric: {:.3f}, lr: {:.3f}".format(val_metric, self._learning_rate))

                if val_metric <= max_val_metric:
                    last_min_val_metric_update += 1
                else:
                    last_min_val_metric_update = 0
                    max_val_metric = val_metric

                y_preprocessed = np.ones(len(y_train))
                y_preprocessed[y_train[:, 0] == 1] = -1
                
                train_classes_prediction = self._softmax(train_prediction).argmax(axis=-1)
                train_classes_prediction[train_classes_prediction == 0] = -1
                
                current_weights = np.exp(-train_classes_prediction * y_preprocessed)
                current_weights /= current_weights.sum()
        except KeyboardInterrupt:
            ...

        if last_min_val_metric_update > 0:
            self._estimators = self._estimators[:-last_min_val_metric_update]

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predictions = None
        for estimator in self._estimators:
            current_prediction = estimator.predict_proba(X)
            predictions = current_prediction if predictions is None else predictions + current_prediction

        return self._softmax(predictions)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=-1)
