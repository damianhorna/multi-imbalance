from unittest import TestCase

import pandas as pd
import sklearn
import numpy as np

from multi_imbalance.classifiers.bracid.bracid import BRACID
from tests.classifiers.bracid.classes_ import _0, _1, _2


class TestCv(TestCase):
    """Tests cv() in utils.py"""

    def test_cv(self):
        """Tests that cross-validation is performed correctly"""
        dataset = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75], "C": [3, 2, 1, 0.5, 3, 2], "Class": [_0, _0, _1, _2, _1, _2]})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        class_col_name = "Class"
        k = 3
        bracid = BRACID(k=k, minority_class=-1)
        classes = [_0, _1, _2]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})

        folds = 2
        seed = 13
        micro_f1, class_f1, true_fold, predicted_fold = bracid.cv_multiclass(
            dataset, k, class_col_name, min_max, classes, folds=folds, seed=seed
        )

        # Double check: convert (3,) to 1d array to see if results are correct
        true_fold = true_fold.reshape(-1)
        predicted_fold = predicted_fold.reshape(-1)
        micro_f1_fold = sklearn.metrics.f1_score(true_fold, predicted_fold, labels=classes, average="micro")
        classwise_f1_fold = sklearn.metrics.f1_score(true_fold, predicted_fold, labels=classes, average=None)
        correct_micro = 1 / 6
        correct_classwise_f1 = np.array([0, 0, 1 / 3])
        np.testing.assert_array_almost_equal(correct_classwise_f1, class_f1)
        self.assertAlmostEqual(correct_micro, micro_f1)
        self.assertAlmostEqual(micro_f1_fold, correct_micro)
        np.testing.assert_array_equal(correct_classwise_f1, classwise_f1_fold)
