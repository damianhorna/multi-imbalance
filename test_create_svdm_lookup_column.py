from unittest import TestCase
from collections import Counter

import pandas as pd
from pandas.api.types import is_numeric_dtype

from scripts.vars import CONDITIONAL
from scripts.utils import create_svdm_lookup_column


class TestCreateLookupMatrix(TestCase):
    """Tests create_svdm_lookup_column() from utils.py"""

    def test_lookup_empty(self):
        """
        Test that an empty dictionary is created if no nominal feature exists
        """
        df = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]})
        lookup = {}
        for i, _ in enumerate(df):
            col = df.iloc[:, i]
            if not is_numeric_dtype(col):
                lookup[i] = create_svdm_lookup_column(df, col, "")
        self.assertTrue(lookup == dict())

    def test_lookup_single_feature(self):
        """
        Test that a correct matrix is created if one feature is nominal
        """
        df = pd.DataFrame({"A": ["high", "low", "high", "low", "low", "high"], "B": [3, 2, 1, 1, 1, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        lookup = {}
        class_col_name = "Class"
        for col_name in df:
            if col_name != class_col_name:
                col = df[col_name]
                if not is_numeric_dtype(col):
                    lookup[col_name] = create_svdm_lookup_column(df, col, class_col_name)
        correct =\
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
        self.assertTrue(lookup == correct)

    def test_lookup_multiple_features(self):
        """
        Test that a correct matrix is created if two features are nominal
        """
        df = pd.DataFrame({"A": ["high", "low", "high", "low", "low", "high"], "B": ["A", "A", "B", "C", "A", "A"],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        lookup = {}
        class_col_name = "Class"
        for col_name in df:
            if col_name != class_col_name:
                col = df[col_name]
                if not is_numeric_dtype(col):
                    lookup[col_name] = create_svdm_lookup_column(df, col, class_col_name)
        correct = \
            {"A": {
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
                    }},
             "B": {
                 'A': 4,
                 'B': 1,
                 'C': 1,
                 CONDITIONAL:
                     {
                         'A':
                             Counter({
                              'apple': 2,
                              'banana': 2
                             }),
                         'B': Counter({
                             'banana': 1
                         }),
                         'C': Counter({'banana': 1
                                       })
                      }
             }
             }
        self.assertTrue(lookup == correct)
