import numpy as np


class SpookyCriterion:
    def __init__(self, **kwargs):
        ...

    def __call__(self, y: np.ndarray, predictions: np.ndarray) -> float:
        raise NotImplementedError()
    
    def predict(self, y: np.ndarray) -> float:
        raise NotImplementedError()
    
    def grad_output(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
