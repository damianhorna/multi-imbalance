from unittest import TestCase

import pandas as pd

from multi_imbalance.resampling.bracid.bracid import BRACID
import multi_imbalance.resampling.bracid.vars as my_vars
from tests.resampling.bracid.classes_ import _0, _1, _2


class TestExtractRulesAndTrainAndPredictMulticlass(TestCase):
    """Checks test_extract_rules_and_train_and_predict_multiclass in utils.py"""

    def test_extract_rules_and_train_and_predict_multiclass(self):
        """Tests that multiclass classification is performed correctly"""
        train_set = pd.DataFrame({"B": [1, 1, 4, 1.5, 0.5, 0.75],
                                  "C": [3, 2, 1, .5, 3, 2],
                                  "Class": [_0, _0, _1, _2, _1, _2]})
        test_set = pd.DataFrame({"B": [4.1, 1, 5.4, 0.15, 0.05, 0.075],
                                 "C": [0.3, 2, 0.1, .4, 0.3, 5],
                                 "Class": ["", "", "", "", "", ""]})
        bracid = BRACID()
        class_col_name = "Class"
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})

        k = 3
        rules, preds_df = \
            bracid.extract_rules_and_train_and_predict_multiclass(train_set, test_set, min_max, class_col_name, k)
        correct_preds = pd.DataFrame({
            my_vars.PREDICTED_LABEL: [_1, _0, _1, _1, _1, _1],
            my_vars.PREDICTION_CONFIDENCE: [0.7, 0.59667, 0.7, 0.4, 0.4, 0.4]
        })
        pd.testing.assert_series_equal(correct_preds[my_vars.PREDICTION_CONFIDENCE], preds_df[my_vars.PREDICTION_CONFIDENCE], check_less_precise=4)
        pd.testing.assert_series_equal(correct_preds[my_vars.PREDICTED_LABEL], preds_df[my_vars.PREDICTED_LABEL])
