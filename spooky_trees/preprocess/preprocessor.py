import numpy as np
import pandas as pd


class Preprocessor:
    def fit(self, X: pd.DataFrame):
        raise NotImplementedError()
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError()
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
