import typing as T

import numpy as np


class SpookyCriterion:
    def __call__(self, y: np.ndarray) -> float:
        raise NotImplementedError()
    
    def rolling(self, y: np.ndarray) -> T.List[T.Tuple[float, float]]:
        raise NotImplementedError()
