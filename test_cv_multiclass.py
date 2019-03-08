from unittest import TestCase
from collections import Counter

import pandas as pd
import sklearn
import numpy as np

from scripts.utils import cv_multiclass
import scripts.vars as my_vars


class TestCv(TestCase):
    """Tests cv() in utils.py"""

    def test_cv(self):
        """Tests that cross-validation is performed correctly"""
        dataset = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": ["apple", "apple", "banana", "orange", "banana", "orange"]})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        "high": 2,
                        "low": 4,
                        my_vars.CONDITIONAL:
                            {
                                "high":
                                    Counter({
                                        "banana": 1,
                                        "orange": 1
                                    }),
                                "low":
                                    Counter({
                                        "banana": 1,
                                        "apple": 2,
                                        "orange": 1
                                    })
                            }
                    }
            }
        classes = ["apple", "banana", "orange"]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})

        k = 3
        folds = 2
        seed = 13
        micro_f1, class_f1, true_fold, predicted_fold = cv_multiclass(dataset, k, class_col_name, lookup, min_max,
                                                                      classes, folds=folds, seed=seed)

        # Double check: convert (3,) to 1d array to see if results are correct
        true_fold = true_fold.reshape(-1)
        predicted_fold = predicted_fold.reshape(-1)
        micro_f1_fold = sklearn.metrics.f1_score(true_fold, predicted_fold, classes, average="micro")
        classwise_f1_fold = sklearn.metrics.f1_score(true_fold, predicted_fold, classes, average=None)
        correct_micro = 1/6
        correct_classwise_f1 = np.array([0.4, 0, 0])
        self.assertTrue(np.array_equal(correct_classwise_f1, class_f1))
        self.assertTrue(correct_micro == micro_f1)
        self.assertTrue(micro_f1_fold == correct_micro)
        self.assertTrue(np.array_equal(correct_classwise_f1, classwise_f1_fold))
