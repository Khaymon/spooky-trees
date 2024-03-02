from copy import deepcopy
from tqdm import tqdm
import typing as T

import numpy as np

from spooky_trees.criterions import SpookyCriterion, SpookyMSE
from spooky_trees.metrics import cross_entropy, f1_score
from spooky_trees.model_selection import train_test_split
from .spooky_tree import SpookyTree


class GradSpookyBoost:
    def __init__(
        self,
        n_estimators: int,
        learning_rate: float = 1e-1,
        val_size: float = 0.05,
        learning_rate_decay: float = 0.95,
        reduce_lr_on_plateu: bool = False,
        reduce_lr_on_plateu_patience: int = -1,
        early_stop: bool = False,
        early_stop_patience: int = -1,
        val_metric: T.Callable[[np.ndarray, np.ndarray], float] = f1_score,
        **kwargs
    ):
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._val_size = val_size
        self._learning_rate_decay = learning_rate_decay
        self._reduce_lr_on_plateu = reduce_lr_on_plateu

        if early_stop:
            assert early_stop_patience >= 0
            self._early_stop_patience = early_stop_patience

        if reduce_lr_on_plateu:
            assert reduce_lr_on_plateu_patience >= 0
            assert reduce_lr_on_plateu_patience < early_stop_patience

            self._reduce_lr_on_plateu_patience = reduce_lr_on_plateu_patience

        self._val_metric = val_metric

        self.spooky_tree_params = kwargs

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        return np.exp(logits) / (np.sum(np.exp(logits), axis=1, keepdims=True))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradSpookyBoost":
        assert y.ndim == 2

        X_train, X_val, y_train, y_val = train_test_split(X, y, self._val_size)

        current_target = y_train
        current_criterion: SpookyCriterion = self.spooky_tree_params["criterion"](**self.spooky_tree_params)
        self._estimators = []

        train_prediction = np.zeros_like(y_train)
        val_prediction = np.zeros_like(y_val)

        max_val_metric = -1e9
        last_min_val_metric_update = 0

        progress_bar = tqdm(range(self._n_estimators))
        try:
            for idx in progress_bar:
                current_tree_params = deepcopy(self.spooky_tree_params)
                if idx > 0:
                    current_tree_params["criterion"] = SpookyMSE
                
                current_estimator = SpookyTree(**current_tree_params).fit(X_train, current_target)
                self._estimators.append(current_estimator)

                current_train_prediction = current_estimator.predict_proba(X_train)
                current_val_prediction = current_estimator.predict_proba(X_val)
                
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
                
                if last_min_val_metric_update > self._reduce_lr_on_plateu_patience:
                    self._learning_rate *= self._learning_rate_decay
                if last_min_val_metric_update > self._early_stop_patience:
                    break

                current_target = -self._learning_rate * current_criterion.grad_output(y_train, train_prediction)
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
