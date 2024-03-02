import numpy as np


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y_true) == len(y_pred)

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(axis=1)

    if y_true.ndim == 2:
        assert np.all(y_true.sum(axis=1) == 1)
        y_true = np.argmax(y_true, axis=1)
    elif y_true.ndim > 2:
        raise NotImplementedError()
    
    true_positives = np.sum((y_true == y_pred) & (y_pred == 1))
    false_positives = np.sum((y_true != y_pred) & (y_pred == 1))
    false_negatives = np.sum((y_true != y_pred) & (y_pred == 0))
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    return 2 * precision * recall / (precision + recall)

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))
