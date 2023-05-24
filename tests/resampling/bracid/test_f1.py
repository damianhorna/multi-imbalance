from unittest import TestCase

from multi_imbalance.resampling.bracid.bracid import BRACID, ConfusionMatrix
import multi_imbalance.resampling.bracid.vars as my_vars
from tests.resampling.bracid.classes_ import _0, _1


class TestF1(TestCase):
    """Test f1() in utils.py"""

    # def test_f1_exception(self):
    #     """Tests if exception is thrown in case of unequal list lengths"""
    #     predicted = ["a", "b", "a"]
    #     true = ["a", "a"]
    #     positive = "a"
    #     self.assertRaises(Exception, f1, predicted, true, positive)

    def test_f1_high_recall(self):
        """Tests if F1 is computed correctly"""
        bracid = BRACID()
        bracid.conf_matrix = ConfusionMatrix(
            TP={1, 2},
            TN={7},
            FP={3, 4, 5, 6},
            FN={0},
        )
        score = bracid.f1(bracid.conf_matrix)
        # Assume that positive class is "a"
        correct = 2*1/3*2/3
        self.assertEqual(score, correct)

    def test_f1_high_precision(self):
        """Tests if F1 is computed correctly"""
        bracid = BRACID()
        bracid.conf_matrix = ConfusionMatrix(
            TP={0, 1, 6},
            TN={},
            FP={4},
            FN={2, 3, 5, 7},
        )
        # Assume that positive class is "a"
        correct = 2*3/7*3/4 / (3/7 + 3/4)
        score = bracid.f1(bracid.conf_matrix)
        self.assertEqual(score, correct)

    def test_f1_zero(self):
        """Tests if F1 is 0 if precision and recall are 0"""
        bracid = BRACID()
        bracid.conf_matrix = ConfusionMatrix()
        # Assume that positive class is "a"
        correct = 0
        score = bracid.f1(bracid.conf_matrix)
        self.assertEqual(score, correct)

    def test_f1_none(self):
        """Tests if F1 is 0 if confusion matrix is None"""
        bracid = BRACID()
        bracid.conf_matrix = None
        # Assume that positive class is "a"
        correct = 0
        score = bracid.f1(bracid.conf_matrix)
        self.assertEqual(score, correct)





