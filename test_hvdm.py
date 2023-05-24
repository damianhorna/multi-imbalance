from unittest import TestCase
from collections import Counter

import pandas as pd
import numpy as np

from scripts.vars import CONDITIONAL
from scripts.bracid import BRACID


class TestHvdm(TestCase):
    """Test hvdm() in utils.py"""

    def test_hvdm_numeric_nominal(self):
        """Tests what happens if input has a numeric and a nominal feature"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
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
        correct = pd.DataFrame({"A": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0], "B": [0, 0, 0.09, 0.0025, 0.0025, 0.000625]})
        correct["dist"] = correct.select_dtypes(float).sum(1)
        correct = correct.sort_values("dist", ascending=True)
        rule = pd.Series({"A": "high", "B": (1, 1), "Class": _1})
        classes = [_0, _1]
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        dist = bracid.hvdm(df, rule, lookup, classes, min_max, class_col_name)
        # Due to floating point precision, use approximate comparison
        np.testing.assert_allclose(correct["A"], dist["A"])
        np.testing.assert_allclose(correct["B"], dist["B"])
        np.testing.assert_allclose(correct["dist"], dist["dist"])

    def test_hvdm_numeric(self):
        """Tests what happens if input has only one type of input, namely a numeric feature"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
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
        correct = pd.DataFrame({"B": [0, 0, 0.09, 0.0025, 0.0025, 0.000625]})
        correct["dist"] = correct.select_dtypes(float).sum(1)
        correct = correct.sort_values("dist", ascending=True)
        rule = pd.Series({"B": (1, 1), "Class": _1})
        classes = [_0, _1]
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        dist = bracid.hvdm(df, rule, lookup, classes, min_max, class_col_name)
        # Due to floating point precision, use approximate comparison
        np.testing.assert_allclose(correct["B"], dist["B"])
        np.testing.assert_allclose(correct["dist"], dist["dist"])

    def test_hvdm_nominal(self):
        """Tests what happens if input has only one type of input, namely a nominal feature"""
        bracid = BRACID()
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
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
        correct = pd.DataFrame({"A": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]})
        correct["dist"] = correct.select_dtypes(float).sum(1)
        correct = correct.sort_values("dist", ascending=True)
        rule = pd.Series({"A": "high", "Class": _1})
        classes = [_0, _1]
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        dist = bracid.hvdm(df, rule, lookup, classes, min_max, class_col_name)
        # Due to floating point precision, use approximate comparison
        np.testing.assert_allclose(correct["A"], dist["A"])
        np.testing.assert_allclose(correct["dist"], dist["dist"])
