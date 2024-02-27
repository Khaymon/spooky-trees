import numpy as np


class SpookyPredicate:
    def __call__(self, X: np.ndarray) -> np.ndarray | bool:
        raise NotImplementedError()
