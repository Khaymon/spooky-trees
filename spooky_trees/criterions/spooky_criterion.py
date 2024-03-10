import numpy as np


class SpookyCriterion:
    def __init__(self, **kwargs):
        ...

    def __call__(self, y: np.ndarray, predictions: np.ndarray, **kwargs) -> float:
        raise NotImplementedError()
    
    def predict(self, y: np.ndarray, **kwargs) -> float:
        raise NotImplementedError()
    
    def grad_output(self, y: np.ndarray, predictions: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError()
