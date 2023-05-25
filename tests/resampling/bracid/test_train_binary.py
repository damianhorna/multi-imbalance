from unittest import TestCase

import pandas as pd

from multi_imbalance.resampling.bracid.bracid import BRACID, Bounds, Support
from tests.resampling.bracid.classes_ import _0, _1


class TestTrain(TestCase):
    """Tests train from utils.py"""

    def test_train(self):
        """Test with numeric and nominal features"""
        bracid = BRACID()
        training_set = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75],
                                     "C": [3, 2, 1, .5, 3, 2],
                                     "Class": [_0, _0, _1, _1, _1, _1]})
        # Use majority class as minority to have multiple neighbors and see if the function works correctly
        minority_label = _1
        class_col_name = "Class"
        rules = {
            2: pd.Series({"B": Bounds(lower=1.25, upper=4.0), "C": Bounds(lower=0.5, upper=1.5),
                          "Class": _1}, name=2),
            6: pd.Series({"B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                          "Class": _1}, name=6),
            5: pd.Series({"B": Bounds(lower=0.75, upper=4.0), "C": Bounds(lower=1.0, upper=2.5),
                          "Class": _1}, name=5),
            0: pd.Series({"B": Bounds(lower=0.5, upper=1.5), "C": Bounds(lower=0.5, upper=3.0),
                          "Class": _0}, name=0),
        }
        model = bracid.train_binary(rules, training_set, minority_label, class_col_name)
        correct_model = {2: Support(minority=1.0, majority=0.0),
                         6: Support(minority=0.6, majority=0.4),
                         5: Support(minority=0.6666666666666666,
                                    majority=0.33333333333333337),
                         0: Support(minority=0.6, majority=0.4)}
        self.assertDictEqual(model, correct_model)
