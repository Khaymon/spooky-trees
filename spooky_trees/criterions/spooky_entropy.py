import numpy as np
import pandas as pd
import typing as T

from .spooky_criterion import SpookyCriterion


class SpookyEntropy(SpookyCriterion):
    def __call__(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0

        _, counts = np.unique(y, return_counts=True)

        if counts.sum() == 0:
            return 0
        target_probas = counts / counts.sum()

        return np.sum(-target_probas * np.log(target_probas))
