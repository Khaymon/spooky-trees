import numpy as np

from .spooky_criterion import SpookyCriterion


class SpookyMSE(SpookyCriterion):
    def __init__(self, l2_leaf_reg: float = 1e-3, l1_leaf_reg: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        self._l2_leaf_reg = l2_leaf_reg
        self._l1_leaf_reg = l1_leaf_reg

    def __call__(self, y: np.ndarray, predictions: np.ndarray, **kwargs) -> float:
        if len(y) == 0:
            return 0.0

        return np.mean((y - predictions) ** 2) + self._l2_leaf_reg * np.sum(predictions ** 2) + self._l1_leaf_reg * np.sum(np.abs(predictions)) 
    
    def predict(self, y: np.ndarray, **kwargs) -> float:
        if len(y) == 0:
            return 0.0

        return y.mean(axis=0)
    
    def grad_output(self, y: np.ndarray, predictions: np.ndarray, **kwargs) -> np.ndarray:
        return y - predictions
