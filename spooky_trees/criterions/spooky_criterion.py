import pandas as pd


class SpookyCriterion:
    def __call__(self, y: pd.Series) -> float:
        raise NotImplementedError()