from unittest import TestCase
from collections import Counter

import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np

from scripts.vars import CONDITIONAL
from scripts.bracid import BRACID


class TestSvdm(TestCase):
    """Tests svdm() from utils.py"""

    def test_svdm_nan_row(self):
        """Tests that correct svdm is computed if NaNs occur in a row of a column"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["high", np.nan, "high", "low", "low", "high"], "B": [3, 2, 1, 1, 1, 2],
                           "C": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 3,
                        'low': 2,
                        CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        _1: 2,
                                        _0: 1
                                    }),
                                'low':
                                    Counter({
                                        _1: 2
                                    })
                            }
                    }
            }
        rule = pd.Series({"A": "high", "B": (1, 1), "C": "bla", "Class": _1})
        classes = [_0, _1]
        correct = [pd.Series([0.0, 1.0, 0.0, 2/3*2/3, 2/3*2/3, 0.0], name="A"),
                   pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], name="A")]
        j = 0
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_string_dtype(col):
                dist = bracid.svdm(col, rule, lookup, classes)
                if j == 0:
                    np.testing.assert_allclose(correct[0], dist)
                else:
                    pd.testing.assert_series_equal(dist, correct[j])
                j += 1

    def test_svdm_nan_rule(self):
        """Tests that correct svdm is computed if NaNs occur in a rule"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["high", np.nan, "high", "low", "low", "high"], "B": [3, 2, 1, 1, 1, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 3,
                        'low': 3,
                        CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        _1: 2,
                                        _0: 1
                                    }),
                                'low':
                                    Counter({
                                        _1: 2,
                                        _0: 1
                                    })
                            }
                    }
            }
        rule = pd.Series({"A": np.NaN, "B": (1, 1), "Class": _1})
        classes = [_0, _1]
        correct = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], name="A")
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_string_dtype(col):
                dist = bracid.svdm(col, rule, lookup, classes)
                pd.testing.assert_series_equal(dist, correct, check_names=False)

    def test_svdm_single_feature(self):
        """Tests that correct svdm is computed for 1 nominal feature"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [3, 2, 1, 1, 1, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 2,
                        'low': 4,
                        CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        _1: 2
                                    }),
                                'low':
                                    Counter({
                                        _1: 2,
                                        _0: 2
                                    })
                            }
                    }
            }
        correct = pd.Series({2: 0.0, 3: 1.0, 4: 1.0, 5: 0.0})
        rule = pd.Series({"A": "high", "B": (1, 1), "Class": _1})
        dist = None
        classes = [_0, _1]
        # Only keep rows with the same class label as the rule
        df = df.loc[df[class_col_name] == _1]
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_string_dtype(col):
                dist = bracid.svdm(col, rule, lookup, classes)
        pd.testing.assert_series_equal(dist, correct, check_names=False)

    def test_svdm_single_feature2(self):
        """Tests that correct svdm is computed for 1 nominal feature"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["high", "low", "high", "low", "low", "high"], "B": [3, 2, 1, 1, 1, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 3,
                        'low': 3,
                        CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        _1: 2,
                                        _0: 1
                                    }),
                                'low':
                                    Counter({
                                        _1: 2,
                                        _0: 1
                                    })
                            }
                    }
            }
        correct = pd.Series({2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0})
        rule = pd.Series({"A": "high", "B": (1, 1), "Class": _1})
        dist = None
        classes = [_0, _1]
        # Only keep rows with the same class label as the rule
        df = df.loc[df[class_col_name] == _1]
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_string_dtype(col):
                dist = bracid.svdm(col, rule, lookup, classes)
        pd.testing.assert_series_equal(dist, correct, check_names=False)

    def test_svdm_multiple_features(self):
        """Tests that correct svdm is computed for 2 nominal features"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["high", "low", "high", "low", "low", "high"], "B": ["x", "y", "x", "x", "y", "x"],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 3,
                        'low': 3,
                        CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        _1: 2,
                                        _0: 1
                                    }),
                                'low':
                                    Counter({
                                        _1: 2,
                                        _0: 1
                                    })
                            }
                    },
                "B":
                    {
                        'x': 4,
                        'y': 2,
                        CONDITIONAL:
                            {
                                'x':
                                    Counter({
                                        _1: 3,
                                        _0: 1
                                    }),
                                'y':
                                    Counter({
                                        _1: 1,
                                        _0: 1
                                    })
                            }
                    }
            }
        correct = [pd.Series({2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}),
                   pd.Series({2: 0.0, 3: 0.0, 4: 0.25, 5: 0.0})]
        rule = pd.Series({"A": "high", "B": "x", "Class": _1})
        dists = []
        classes = [_0, _1]
        # Only keep rows with the same class label as the rule
        df = df.loc[df[class_col_name] == _1]
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_string_dtype(col):
                dists.append(bracid.svdm(col, rule, lookup, classes))
            pd.testing.assert_series_equal(dists[i], correct[i], check_names=False)

    def test_svdm_multiple_features_multiple_rules(self):
        """Tests that correct svdm is computed for 2 nominal features with 2 rules"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["high", "low", "high", "low", "low", "high"], "B": ["x", "y", "x", "x", "y", "x"],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 3,
                        'low': 3,
                        CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        _1: 2,
                                        _0: 1
                                    }),
                                'low':
                                    Counter({
                                        _1: 2,
                                        _0: 1
                                    })
                            }
                    },
                "B":
                    {
                        'x': 4,
                        'y': 2,
                        CONDITIONAL:
                            {
                                'x':
                                    Counter({
                                        _1: 3,
                                        _0: 1
                                    }),
                                'y':
                                    Counter({
                                        _1: 1,
                                        _0: 1
                                    })
                            }
                    }
            }
        correct = [pd.Series({2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}),
                   pd.Series({2: 0.0, 3: 0.0, 4: 0.25, 5: 0.0})]
        rules = [pd.Series({"A": "high", "B": "x", "Class": _1}),
                pd.Series({"A": "high", "B": "x", "Class": _1})]
        dists = []
        classes = [_0, _1]
        # Only keep rows with the same class label as the rule
        df = df.loc[df[class_col_name] == _1]
        for rule in rules:
            for i, col_name in enumerate(df):
                if col_name == class_col_name:
                    continue
                col = df[col_name]
                if is_string_dtype(col):
                    dists.append(bracid.svdm(col, rule, lookup, classes))
                pd.testing.assert_series_equal(dists[i], correct[i], check_names=False)
