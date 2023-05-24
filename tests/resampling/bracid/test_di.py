from unittest import TestCase

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

# from scripts.utils import di
from multi_imbalance.resampling.bracid.bracid import BRACID
from tests.resampling.bracid.classes_ import _0, _1

class TestDi(TestCase):
    """Test di() in utils.py"""

    def test_di_nan_row(self):
        """Tests that correct distance is computed if NaNs occur in a row of a column"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["high", np.nan, "high", "low", "low", "high"], "B": [3, 2, 1, np.nan, 0.5, 2],
                           "C": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        rule = pd.Series({"A": "high", "B": (1, 2), "C":(1, np.NaN), "Class": _1})
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}})
        correct = [pd.Series([1/4*1/4, 0.0, 0.0, 1.0, 1/8*1/8, 0.0], name="A"),
                   pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], name="A")]
        j = 0
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_numeric_dtype(col):
                dist = bracid.di(col, rule, min_max)
                pd.testing.assert_series_equal(dist, correct[j], check_names=False)
                j += 1

    def test_di_nan_rule(self):
        """Tests that correct distance is computed if NaNs occur in a rule"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["high", np.nan, "high", "low", "low", "high"], "B": [3, 2, 1, np.nan, 1, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        rule = pd.Series({"A": "high", "B": (np.NaN, 2), "Class": _1})
        min_max = pd.DataFrame({"B": {"min": 1, "max": 2}})
        correct = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], name="A")
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_numeric_dtype(col):
                dist = bracid.di(col, rule, min_max)
                pd.testing.assert_series_equal(dist, correct, check_names=False)

    def test_di_single_feature(self):
        """Tests that correct distance is computed for 1 numeric feature"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["high", np.nan, "high", "low", "low", "high"], "B": [3, 2, 1, .5, 1, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        rule = pd.Series({"A": "high", "B": (1, 2), "Class": _1})
        dist = None
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}})
        correct = pd.Series({0: 0.25*0.25, 1: 0, 2: 0, 3: 0.125*0.125, 4: 0, 5: 0.0})
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_numeric_dtype(col):
                dist = bracid.di(col, rule, min_max)
        pd.testing.assert_series_equal(dist, correct, check_names=False)

    def test_di_multiple_features(self):
        """Tests that correct distance is computed for 2 numeric features"""
        bracid = BRACID()
        df = pd.DataFrame({"A": [1, 1, 4, 1.5, 0.5, 0.75], "B": [3, 2, 1, .5, 3, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        rule = pd.Series({"A": (1, 2), "B": (1, 2), "Class": _1})
        dists = []
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        # Only keep rows with the same class label as the rule
        df = df.loc[df[class_col_name] == _1]
        correct = [pd.Series({2: 0.5*0.5, 3: 0.0, 4: 0.125*0.125, 5: 1/16*1/16}),
                   pd.Series({2: 0, 3: 0.05 * 0.05, 4: 0.1*0.1, 5: 0.0})]
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_numeric_dtype(col):
                dist = bracid.di(col, rule, min_max)
                dists.append(dist)
                pd.testing.assert_series_equal(dists[i], correct[i], check_names=False)

    def test_di_multiple_features_multiple_rules(self):
        """Tests that correct distance is computed for 2 numeric features"""
        bracid = BRACID()
        df = pd.DataFrame({"A": [1, 1, 4, 1.5, 0.5, 0.75], "B": [3, 2, 1, .5, 3, 2],
                           "C": ["a", "b", "c", "d", "e", "f"],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        rules = [pd.Series({"A": (1, 2), "B": (1, 2), "Class": _1}),
                 pd.Series({"A": (1, 2), "B": (1, 2), "Class": _1})]
        dists = []
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        # Only keep rows with the same class label as the rule
        df = df.loc[df[class_col_name] == _1]
        correct = [pd.Series({2: 0.5*0.5, 3: 0.0, 4: 0.125*0.125, 5: 1/16*1/16}),
                   pd.Series({2: 0, 3: 0.05 * 0.05, 4: 0.1*0.1, 5: 0.0}),
                   pd.Series({2: 0.5 * 0.5, 3: 0.0, 4: 0.125 * 0.125, 5: 1 / 16 * 1 / 16}),
                   pd.Series({2: 0, 3: 0.05 * 0.05, 4: 0.1 * 0.1, 5: 0.0})]
        for rule in rules:
            for i, col_name in enumerate(df):
                if col_name == class_col_name:
                    continue
                col = df[col_name]
                if is_numeric_dtype(col):
                    dist = bracid.di(col, rule, min_max)
                    dists.append(dist)
                    pd.testing.assert_series_equal(dists[i], correct[i], check_names=False)
