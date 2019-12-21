import unittest
from unittest.mock import MagicMock, patch

import numpy
from sklearn.tree import tree

from multi_imbalance.ensemble.mrbbagging import MRBBagging

mrbbagging = MRBBagging()


class TestMRBBagging(unittest.TestCase):
    def test__group_data(self):
        x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        y = ["A", "B", "C"]
        classes, grouped_data = mrbbagging._group_data(x, y)
        self.assertEqual(classes, {'A', 'B', 'C'})
        self.assertEqual(grouped_data, {'C': [[[3, 3, 3], 'C']], 'A': [[[1, 1, 1], 'A']], 'B': [[[2, 2, 2], 'B']]})

    def test__group_data_with_none(self):
        x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        y = ["A", None, "C"]
        with self.assertRaises(AssertionError):
            mrbbagging._group_data(x, y)

    def test_fit_with_invalid_labels(self):
        x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        y = ["A", "B"]
        with self.assertRaises(AssertionError):
            mrbbagging.fit(x, y, MagicMock())

    def test_fit_with_invalid_classifier(self):
        x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        y = ["A", "B", "C"]
        with self.assertRaises(AssertionError):
            mrbbagging.fit(x, y, None)


if __name__ == '__main__':
    unittest.main()
