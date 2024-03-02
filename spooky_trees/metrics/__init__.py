import numpy as np


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert len(y_true) == len(y_pred)
    assert y_true.ndim == y_pred.ndim == 1
    
    true_positives = np.sum((y_true == y_pred) & (y_pred == 1))
    false_positives = np.sum((y_true != y_pred) & (y_pred == 1))
    false_negatives = np.sum((y_true != y_pred) & (y_pred == 0))
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    return 2 * precision * recall / (precision + recall)
