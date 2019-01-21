from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import cv, Bounds, Support, Predictions
import scripts.vars as my_vars


class TestCv(TestCase):
    """Tests cv() in utils.py"""

    def test_cv(self):
        """Tests that cross-validation is performed correctly"""
        dataset = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        minority_label = "banana"
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 2,
                        'low': 4,
                        my_vars.CONDITIONAL:
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
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})

        k = 3
        folds = 2
        cv(dataset, k, class_col_name, lookup, min_max, classes, minority_label, folds=folds)
        correct_model = {2: Support(minority=1.0, majority=0.0), 6: Support(minority=0.5, majority=0.5),
                         5: Support(minority=1.0, majority=0.0), 0: Support(minority=0.5, majority=0.5)}
        self.fail()
