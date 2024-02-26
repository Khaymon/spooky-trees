import pandas as pd
import typing as T


class SpookyCriterion:
    def __call__(self, y: pd.Series) -> float:
        raise NotImplementedError()
    
    def rolling(self, y: pd.Series) -> T.List[T.Tuple[float, float]]:
        raise NotImplementedError()
