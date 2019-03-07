from unittest import TestCase
from collections import Counter

import pandas as pd

from scripts.utils import extract_rules_and_train_and_predict_multiclass
import scripts.vars as my_vars


class TestExtractRulesAndTrainAndPredictMulticlass(TestCase):
    """Checks test_extract_rules_and_train_and_predict_multiclass in utils.py"""

    def test_extract_rules_and_train_and_predict_multiclass(self):
        """Tests that multiclass classification is performed correctly"""
        train_set = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                                  "C": [3, 2, 1, .5, 3, 2],
                                  "Class": ["apple", "apple", "banana", "orange", "banana", "orange"]})
        test_set = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"],
                                 "B": [4.1, 1, 5.4, 0.15, 0.05, 0.075],
                                 "C": [0.3, 2, 0.1, .4, 0.3, 5],
                                 "Class": ["", "", "", "", "", ""]})
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
        min_max = pd.DataFrame({"B": {"min": 1, "max": 5}, "C": {"min": 1, "max": 11}})

        k = 3
        rules, preds_dict, preds_df = \
            extract_rules_and_train_and_predict_multiclass(train_set, test_set, lookup, min_max, class_col_name, k)
