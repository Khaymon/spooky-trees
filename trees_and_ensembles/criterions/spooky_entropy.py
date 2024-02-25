import numpy as np
import pandas as pd

from .spooky_criterion import SpookyCriterion


class SpookyEntropy(SpookyCriterion):
    def __call__(self, y: pd.Series, prediction) -> float:
        if len(y) == 0:
            return 0

        target_counts = y.value_counts()
        target_probas = target_counts / len(y)

        return np.sum(-target_probas * np.log(target_probas))
