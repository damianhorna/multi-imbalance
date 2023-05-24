from unittest import TestCase
from collections import Counter

import pandas as pd
from pandas.api.types import is_numeric_dtype

from scripts.vars import CONDITIONAL
# from scripts.utils import create_svdm_lookup_column
from scripts.bracid import BRACID


class TestCreateLookupMatrix(TestCase):
    """Tests create_svdm_lookup_column() from utils.py"""

    def test_lookup_empty(self):
        """
        Test that an empty dictionary is created if no nominal feature exists
        """
        bracid = BRACID()
        df = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]})
        lookup = {}
        for i, _ in enumerate(df):
            col = df.iloc[:, i]
            if not is_numeric_dtype(col):
                lookup[i] = bracid.create_svdm_lookup_column(df, col, "")
        self.assertTrue(lookup == dict())

    def test_lookup_single_feature(self):
        """
        Test that a correct matrix is created if one feature is nominal
        """
        bracid = BRACID()
        df = pd.DataFrame({"A": ["high", "low", "high", "low", "low", "high"], "B": [3, 2, 1, 1, 1, 2],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        lookup = {}
        class_col_name = "Class"
        for col_name in df:
            if col_name != class_col_name:
                col = df[col_name]
                if not is_numeric_dtype(col):
                    lookup[col_name] = bracid.create_svdm_lookup_column(df, col, class_col_name)
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
        self.assertEqual(lookup, correct)

    def test_lookup_multiple_features(self):
        """
        Test that a correct matrix is created if two features are nominal
        """
        bracid = BRACID()
        df = pd.DataFrame({"A": ["high", "low", "high", "low", "low", "high"], "B": ["A", "A", "B", "C", "A", "A"],
                           "Class": [_0, _0, _1, _1, _1, _1]})
        lookup = {}
        class_col_name = "Class"
        for col_name in df:
            if col_name != class_col_name:
                col = df[col_name]
                if not is_numeric_dtype(col):
                    lookup[col_name] = bracid.create_svdm_lookup_column(df, col, class_col_name)
        correct = \
            {"A": {
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
                    }},
             "B": {
                 'A': 4,
                 'B': 1,
                 'C': 1,
                 CONDITIONAL:
                     {
                         'A':
                             Counter({
                              _0: 2,
                              _1: 2
                             }),
                         'B': Counter({
                             _1: 1
                         }),
                         'C': Counter({_1: 1
                                       })
                      }
             }
             }
        self.assertEqual(lookup, correct)
