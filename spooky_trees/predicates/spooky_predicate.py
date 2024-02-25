import pandas as pd


class SpookyPredicate:
    def __call__(self, X: pd.DataFrame | pd.Series) -> bool:
        raise NotImplementedError()
