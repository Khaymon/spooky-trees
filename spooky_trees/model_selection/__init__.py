import typing as T

import numpy as np
import random


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: T.Union[int, float] = 0.1,
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert len(X) == len(y)
    
    if isinstance(test_size, float):
        assert 0 <= test_size <= 1, "Float value of test size should be between 0 and 1"
        test_size = int(len(X) * test_size)
    elif isinstance(test_size, int):
        assert test_size <= len(y)
    else:
        raise ValueError(f"Test size argument should have int or float type")

    test_ids = np.random.choice(np.arange(len(X)), size=test_size, replace=False)
    train_ids = list(set(np.arange(len(X))).difference(set(test_ids)))
    
    X_train, y_train = X[train_ids], y[train_ids]
    X_test, y_test = X[test_ids], y[test_ids]
    
    return X_train, X_test, y_train, y_test


class KFold:
    def __init__(self, n_splits: int = 3):
        self._n_splits = n_splits

    def split(self, size: int):
        assert size >= self._n_splits, "Size of the dataset must be no less than number of splits"

        self._ids = np.arange(size)
        random.shuffle(self._ids)
        
        self._split_idx = 0
        split_size = len(self._ids) // self._n_splits
        if len(self._ids) % self._n_splits > 0:
            split_size += 1
        for split_idx in range(self._n_splits):
            test_ids = self._ids[split_size * split_idx:split_size * (split_idx + 1)]
            train_ids = np.array(list(set(self._ids).difference(test_ids)))

            yield train_ids, test_ids
