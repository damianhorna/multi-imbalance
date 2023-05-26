from unittest import TestCase

import pandas as pd
import numpy as np

from multi_imbalance.classifiers.bracid.bracid import BRACID
from tests.classifiers.bracid.classes_ import _0, _1


class TestCv(TestCase):
    """Tests cv() in utils.py"""

    def test_cv(self):
        """Tests that cross-validation is performed correctly"""
        minority_label = _1
        k = 3
        bracid = BRACID(k=k, minority_class=minority_label)
        dataset = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75],
                                "C": [3, 2, 1, .5, 3, 2],
                                "Class": [_0, _0, _1, _1, _1, _1]})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        class_col_name = "Class"
        classes = [_0, _1]
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})

        folds = 2
        seed = 135
        micro_f1, classwise_f1 = bracid.cv_binary(dataset, k, class_col_name, min_max, classes, minority_label,
                                           folds=folds, seed=seed)

        correct_micro = 1/6
        correct_classwise = np.array([0, 0.2857])
        np.testing.assert_array_almost_equal(correct_classwise, classwise_f1, decimal=4)
        self.assertEqual(micro_f1, correct_micro)
