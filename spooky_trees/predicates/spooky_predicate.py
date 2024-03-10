import typing as T

import numpy as np


class SpookyPredicate:
    def __call__(self, X: np.ndarray) -> T.Union[np.ndarray, bool]:
        raise NotImplementedError()
