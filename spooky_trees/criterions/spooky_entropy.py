import numpy as np
import pandas as pd
import typing as T

from .spooky_criterion import SpookyCriterion


class SpookyEntropy(SpookyCriterion):
    def _from_counts(self, counts: pd.Series, eps: float = 1e-9) -> float:
        if counts.sum() == 0:
            return 0
        target_probas = counts / counts.sum()

        return np.sum(-target_probas * np.log(target_probas + eps))

    def __call__(self, y: pd.Series) -> float:
        if len(y) == 0:
            return 0
        return self._from_counts(y.value_counts())
    
    def rolling(self, y: pd.Series) -> T.List[float]:
        right_value_counts = y.value_counts()
        left_value_counts = right_value_counts.copy()
        left_value_counts[:] = 0

        entropy_values = []
        for idx in range(len(y)):
            if idx > 0:
                right_value_counts.loc[y.iloc[idx - 1]] -= 1
                left_value_counts.loc[y.iloc[idx - 1]] += 1

            left_entropy = self._from_counts(left_value_counts)
            right_entropy = self._from_counts(right_value_counts)

            entropy_values.append(left_entropy * idx + right_entropy * (len(y) - idx))

        return entropy_values
