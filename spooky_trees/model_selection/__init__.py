import typing as T

import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: int | float = 0.1,
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
