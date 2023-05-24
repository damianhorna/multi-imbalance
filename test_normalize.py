from unittest import TestCase

import pandas as pd
from pandas.api.types import is_numeric_dtype

from scripts.utils import normalize_dataframe, normalize_series


class TestNormalize(TestCase):
    """Tests normalize_dataframe() and normalize_series() from utils.py"""

    def test_normalize_ints_dataframe(self):
        """
        Test that normalization of integers works applied to the whole dataset
        """
        df = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]})
        df = normalize_dataframe(df)
        assert(df.shape == (3, 2))
        for col_name in df:
            if col_name == "A":
                self.assertTrue(df["A"].equals(pd.Series([0.0, 0.5, 1.0])))
            else:
                self.assertTrue(df["B"].equals(pd.Series([1.0, 0.5, 0.0])))

    def test_normalize_floats_dataframe(self):
        """
        Test that normalization of floats works applied to the whole dataset
        """
        df = pd.DataFrame({"A": [1.4, 2.4, 3.4], "B": [3.4, 2.4, 1.4]})
        df = normalize_dataframe(df)
        assert(df.shape == (3, 2))
        for col_idx, _ in enumerate(df):
            if col_idx == 0:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([0.0, 0.5, 1.0])))
            else:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([1.0, 0.5, 0.0])))

    def test_normalize_nominal_dataframe(self):
        """
        Test that normalization is applied only to columns with numeric values applied to the whole dataset
        """
        df = pd.DataFrame({"A": [1, 2, 3], "B": [3.4, 2.4, 1.4], "C": ["A", "B", "C"]})
        df = normalize_dataframe(df)
        assert (df.shape == (3, 3))
        for col_idx, _ in enumerate(df):
            if col_idx == 0:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([0.0, 0.5, 1.0])))
            elif col_idx == 1:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([1.0, 0.5, 0.0])))
            else:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series(["A", "B", "C"])))

    def test_normalize_ints_series(self):
        """
        Test that normalization of integers works applied to the whole dataset columnwise
        """
        df = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]})
        assert(df.shape == (3, 2))

        for col_name in df:
            col = df[col_name]
            if is_numeric_dtype(col):
                df[col_name] = normalize_series(col)
        for col_name in df.columns:
            if col_name == "A":
                self.assertTrue(df["A"].equals(pd.Series([0.0, 0.5, 1.0])))
            else:
                self.assertTrue(df["B"].equals(pd.Series([1.0, 0.5, 0.0])))

    def test_normalize_floats_series(self):
        """
        Test that normalization of floats works applied to the whole dataset columnwise
        """
        df = pd.DataFrame({"A": [1.4, 2.4, 3.4], "B": [3.4, 2.4, 1.4]})
        df = normalize_dataframe(df)
        assert(df.shape == (3, 2))
        for col_idx, _ in enumerate(df):
            if col_idx == 0:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([0.0, 0.5, 1.0])))
            else:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([1.0, 0.5, 0.0])))

    def test_normalize_nominal_series(self):
        """
        Test that normalization is applied only to columns with numeric values applied to the whole dataset columnwise
        """
        df = pd.DataFrame({"A": [1, 2, 3], "B": [3.4, 2.4, 1.4], "C": ["A", "B", "C"]})
        df = normalize_dataframe(df)
        assert (df.shape == (3, 3))
        for col_idx, _ in enumerate(df):
            if col_idx == 0:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([0.0, 0.5, 1.0])))
            elif col_idx == 1:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series([1.0, 0.5, 0.0])))
            else:
                self.assertTrue(df.iloc[:, col_idx].equals(pd.Series(["A", "B", "C"])))
