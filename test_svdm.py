from unittest import TestCase
from collections import Counter

import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np

from scripts.vars import CONDITIONAL
from scripts.utils import svdm


class TestSvdm(TestCase):
    """Tests svdm() from utils.py"""

    def test_svdm_nan_row(self):
        """Tests that correct svdm is computed if NaNs occur in a row of a column"""
        df = pd.DataFrame({"A": ["high", np.nan, "high", "low", "low", "high"], "B": [3, 2, 1, 1, 1, 2],
                           "C": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
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
                                        'banana': 2,
                                        'apple': 1
                                    }),
                                'low':
                                    Counter({
                                        'banana': 2
                                    })
                            }
                    }
            }
        rule = pd.Series({"A": "high", "B": (1, 1), "C": "bla", "Class": "banana"})
        classes = ["apple", "banana"]
        correct = [pd.Series([0.0, 1.0, 0.0, 2/3*2/3, 2/3*2/3, 0.0], name="A"),
                   pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], name="A")]
        j = 0
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_string_dtype(col):
                dist = svdm(col, rule, lookup, classes)
                if j == 0:
                    self.assertTrue(np.allclose(correct[0], dist))
                else:
                    self.assertTrue(dist.equals(correct[j]))
                j += 1

    def test_svdm_nan_rule(self):
        """Tests that correct svdm is computed if NaNs occur in a rule"""
        df = pd.DataFrame({"A": ["high", np.nan, "high", "low", "low", "high"], "B": [3, 2, 1, 1, 1, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
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
                                        'banana': 2,
                                        'apple': 1
                                    }),
                                'low':
                                    Counter({
                                        'banana': 2,
                                        'apple': 1
                                    })
                            }
                    }
            }
        rule = pd.Series({"A": np.NaN, "B": (1, 1), "Class": "banana"})
        classes = ["apple", "banana"]
        correct = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], name="A")
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_string_dtype(col):
                dist = svdm(col, rule, lookup, classes)
                self.assertTrue(dist.equals(correct))

    def test_svdm_single_feature(self):
        """Tests that correct svdm is computed for 1 nominal feature"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [3, 2, 1, 1, 1, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
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
                                        'banana': 2
                                    }),
                                'low':
                                    Counter({
                                        'banana': 2,
                                        'apple': 2
                                    })
                            }
                    }
            }
        correct = pd.Series({2: 0.0, 3: 1.0, 4: 1.0, 5: 0.0})
        rule = pd.Series({"A": "high", "B": (1, 1), "Class": "banana"})
        dist = None
        classes = ["apple", "banana"]
        # Only keep rows with the same class label as the rule
        df = df.loc[df[class_col_name] == "banana"]
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_string_dtype(col):
                dist = svdm(col, rule, lookup, classes)
        self.assertTrue(dist.equals(correct))

    def test_svdm_single_feature2(self):
        """Tests that correct svdm is computed for 1 nominal feature"""
        df = pd.DataFrame({"A": ["high", "low", "high", "low", "low", "high"], "B": [3, 2, 1, 1, 1, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
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
                                        'banana': 2,
                                        'apple': 1
                                    }),
                                'low':
                                    Counter({
                                        'banana': 2,
                                        'apple': 1
                                    })
                            }
                    }
            }
        correct = pd.Series({2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0})
        rule = pd.Series({"A": "high", "B": (1, 1), "Class": "banana"})
        dist = None
        classes = ["apple", "banana"]
        # Only keep rows with the same class label as the rule
        df = df.loc[df[class_col_name] == "banana"]
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_string_dtype(col):
                dist = svdm(col, rule, lookup, classes)
        self.assertTrue(dist.equals(correct))

    def test_svdm_multiple_features(self):
        """Tests that correct svdm is computed for 2 nominal features"""
        df = pd.DataFrame({"A": ["high", "low", "high", "low", "low", "high"], "B": ["x", "y", "x", "x", "y", "x"],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
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
                                        'banana': 2,
                                        'apple': 1
                                    }),
                                'low':
                                    Counter({
                                        'banana': 2,
                                        'apple': 1
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
                                        'banana': 3,
                                        'apple': 1
                                    }),
                                'y':
                                    Counter({
                                        'banana': 1,
                                        'apple': 1
                                    })
                            }
                    }
            }
        correct = [pd.Series({2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}),
                   pd.Series({2: 0.0, 3: 0.0, 4: 0.25, 5: 0.0})]
        rule = pd.Series({"A": "high", "B": "x", "Class": "banana"})
        dists = []
        classes = ["apple", "banana"]
        # Only keep rows with the same class label as the rule
        df = df.loc[df[class_col_name] == "banana"]
        for i, col_name in enumerate(df):
            if col_name == class_col_name:
                continue
            col = df[col_name]
            if is_string_dtype(col):
                dists.append(svdm(col, rule, lookup, classes))
            self.assertTrue(dists[i].equals(correct[i]))

    def test_svdm_multiple_features_multiple_rules(self):
        """Tests that correct svdm is computed for 2 nominal features with 2 rules"""
        df = pd.DataFrame({"A": ["high", "low", "high", "low", "low", "high"], "B": ["x", "y", "x", "x", "y", "x"],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
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
                                        'banana': 2,
                                        'apple': 1
                                    }),
                                'low':
                                    Counter({
                                        'banana': 2,
                                        'apple': 1
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
                                        'banana': 3,
                                        'apple': 1
                                    }),
                                'y':
                                    Counter({
                                        'banana': 1,
                                        'apple': 1
                                    })
                            }
                    }
            }
        correct = [pd.Series({2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}),
                   pd.Series({2: 0.0, 3: 0.0, 4: 0.25, 5: 0.0})]
        rules = [pd.Series({"A": "high", "B": "x", "Class": "banana"}),
                pd.Series({"A": "high", "B": "x", "Class": "banana"})]
        dists = []
        classes = ["apple", "banana"]
        # Only keep rows with the same class label as the rule
        df = df.loc[df[class_col_name] == "banana"]
        for rule in rules:
            for i, col_name in enumerate(df):
                if col_name == class_col_name:
                    continue
                col = df[col_name]
                if is_string_dtype(col):
                    dists.append(svdm(col, rule, lookup, classes))
                self.assertTrue(dists[i].equals(correct[i]))
