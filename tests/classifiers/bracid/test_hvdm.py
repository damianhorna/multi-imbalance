from unittest import TestCase

import pandas as pd
import numpy as np

from multi_imbalance.classifiers.bracid.bracid import hvdm
from tests.classifiers.bracid.classes_ import _0, _1


class TestHvdm(TestCase):
    """Test hvdm() in utils.py"""

    def test_hvdm_numeric(self):
        """Tests what happens if input has only one type of input, namely a numeric feature"""
        df = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75], "C": [3, 2, 1, 0.5, 3, 2], "Class": [_0, _0, _1, _1, _1, _1]})
        class_col_name = "Class"
        correct = pd.DataFrame({"B": [0, 0, 0.09, 0.0025, 0.0025, 0.000625]})
        correct["dist"] = correct.select_dtypes(float).sum(1)
        correct = correct.sort_values("dist", ascending=True)
        rule = pd.Series({"B": (1, 1), "Class": _1})
        classes = [_0, _1]
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        dist = hvdm(df, rule, classes, min_max, class_col_name)
        # Due to floating point precision, use approximate comparison
        np.testing.assert_allclose(correct["B"], dist["B"])
        np.testing.assert_allclose(correct["dist"], dist["dist"])
