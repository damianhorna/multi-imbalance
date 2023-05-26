from unittest import TestCase

import pandas as pd
from pandas.api.types import is_numeric_dtype

from multi_imbalance.classifiers.bracid.bracid import normalize_dataframe, normalize_series


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
                pd.testing.assert_series_equal(df["A"], pd.Series([0.0, 0.5, 1.0]), check_names=False)
            else:
                pd.testing.assert_series_equal(df["B"], pd.Series([1.0, 0.5, 0.0]), check_names=False)

    def test_normalize_floats_dataframe(self):
        """
        Test that normalization of floats works applied to the whole dataset
        """
        df = pd.DataFrame({"A": [1.4, 2.4, 3.4], "B": [3.4, 2.4, 1.4]})
        df = normalize_dataframe(df)
        assert(df.shape == (3, 2))
        for col_idx, _ in enumerate(df):
            if col_idx == 0:
                pd.testing.assert_series_equal(df.iloc[:, col_idx], pd.Series([0.0, 0.5, 1.0]), check_names=False)
            else:
                pd.testing.assert_series_equal(df.iloc[:, col_idx], pd.Series([1.0, 0.5, 0.0]), check_names=False)

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
                pd.testing.assert_series_equal(df["A"], pd.Series([0.0, 0.5, 1.0]), check_names=False)
            else:
                pd.testing.assert_series_equal(df["B"], pd.Series([1.0, 0.5, 0.0]), check_names=False)

    def test_normalize_floats_series(self):
        """
        Test that normalization of floats works applied to the whole dataset columnwise
        """
        df = pd.DataFrame({"A": [1.4, 2.4, 3.4], "B": [3.4, 2.4, 1.4]})
        df = normalize_dataframe(df)
        assert(df.shape == (3, 2))
        for col_idx, _ in enumerate(df):
            if col_idx == 0:
                pd.testing.assert_series_equal(df.iloc[:, col_idx], pd.Series([0.0, 0.5, 1.0]), check_names=False)
            else:
                pd.testing.assert_series_equal(df.iloc[:, col_idx], pd.Series([1.0, 0.5, 0.0]), check_names=False)
