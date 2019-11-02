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
            mrbbagging.fit(x, y, 1, 1, MagicMock())

    def test_fit_with_invalid_classifier(self):
        x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        y = ["A", "B", "C"]
        with self.assertRaises(AssertionError):
            mrbbagging.fit(x, y, 1, 1, None)

    def test__count_votes(self):
        x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        y = ["A", "B", "C"]
        mrbbagging.classifiers = [tree.DecisionTreeClassifier()]
        mrbbagging.classes = {"A", "B", "C"}
        mrbbagging.classifiers[0].fit(x, y)
        mrbbagging.classifier_classes = ["A", "B", "C"]
        expected = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        for id_row, row in enumerate(mrbbagging._count_votes(x)):
            for item_id, item in enumerate(row):
                self.assertEqual(item, expected[id_row][item_id])

    def test__select_classes(self):
        x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        with patch.object(MRBBagging, '_count_votes',
                          return_value=numpy.asarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])):
            self.assertEqual(mrbbagging._select_classes(x), ["A", "B", "C"])


if __name__ == '__main__':
    unittest.main()
