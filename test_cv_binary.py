from unittest import TestCase
from collections import Counter

import pandas as pd
import numpy as np

from scripts.utils import cv_binary
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
        seed = 135
        micro_f1, classwise_f1 = cv_binary(dataset, k, class_col_name, lookup, min_max, classes, minority_label,
                                           folds=folds, seed=seed)

        correct_micro = 1/3
        correct_classwise = np.array([0, 0.5])
        self.assertTrue(np.array_equal(correct_classwise, classwise_f1))
        self.assertTrue(correct_micro == micro_f1)
