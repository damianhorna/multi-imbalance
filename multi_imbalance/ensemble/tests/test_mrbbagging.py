import unittest
from unittest.mock import MagicMock

from sklearn.tree import tree

from multi_imbalance.ensemble.mrbbagging import MRBBagging
import numpy as np

X_train = np.array([
    [0.05837771, 0.57543339],
    [0.06153624, 0.99871925],
    [0.14308529, 0.00681144],
    [0.23401697, 0.21188708],
    [0.2418553, 0.02137086],
    [0.32480534, 0.81547632],
    [0.42478482, 0.31995162],
    [0.50726834, 0.72621157],
    [0.54580968, 0.58025914],
    [0.55748531, 0.71866238],
    [0.69208769, 0.63759459],
    [0.70797377, 0.16348051],
    [0.76410615, 0.70451542],
    [0.81680686, 0.50793884],
    [0.8490789, 0.53826627],
    [0.8847505, 0.96856011],
])
y_train = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0])

X_test = np.array([[0.9287003, 0.97580299],
                   [0.9584236, 0.10536541],
                   [0.01, 0.87666093],
                   [0.97352367, 0.78807909], ])

y_test = np.array([0, 0, 0, 0])


class TestMRBBagging(unittest.TestCase):
    def test_api(self):
        mrbbagging = MRBBagging(1, tree.DecisionTreeClassifier(random_state=0), random_state=0)
        mrbbagging.fit(X_train, y_train)
        y_pred = mrbbagging.predict(X_test)
        assert all(y_pred == y_test)

    def test_api_multiple_trees(self):
        mrbbagging = MRBBagging(5, tree.DecisionTreeClassifier(random_state=0), random_state=0)
        mrbbagging.fit(X_train, y_train)
        y_pred = mrbbagging.predict(X_test)
        assert all(y_pred == y_test)

    def test_api_with_feature_selection(self):
        mrbbagging = MRBBagging(1, tree.DecisionTreeClassifier(random_state=0), feature_selection=True, random_state=0)
        mrbbagging.fit(X_train, y_train)
        y_pred = mrbbagging.predict(X_test)
        assert all(y_pred == y_test)

    def test_api_with_random_feature_selection(self):
        mrbbagging = MRBBagging(1, tree.DecisionTreeClassifier(random_state=0), feature_selection=True, random_fs=True,
                                random_state=0)
        mrbbagging.fit(X_train, y_train)
        y_pred = mrbbagging.predict(X_test)
        assert all(y_pred == y_test)

    def test_api_with_feature_selection_sqrt_features(self):
        mrbbagging = MRBBagging(1, tree.DecisionTreeClassifier(random_state=0), feature_selection=True,
                                half_features=False, random_state=0)
        mrbbagging.fit(X_train, y_train)
        y_pred = mrbbagging.predict(X_test)
        assert all(y_pred == y_test)

    def test__group_data(self):
        mrbbagging = MRBBagging(1, tree.DecisionTreeClassifier())
        x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        y = ["A", "B", "C"]
        classes, grouped_data = mrbbagging._group_data(x, y)
        self.assertEqual(classes, {'A', 'B', 'C'})
        self.assertEqual(grouped_data, {'C': [[[3, 3, 3], 'C']], 'A': [[[1, 1, 1], 'A']], 'B': [[[2, 2, 2], 'B']]})

    def test__group_data_with_none(self):
        mrbbagging = MRBBagging(1, tree.DecisionTreeClassifier())
        x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        y = ["A", None, "C"]
        with self.assertRaises(AssertionError):
            mrbbagging._group_data(x, y)

    def test_fit_with_invalid_labels(self):
        mrbbagging = MRBBagging(1, MagicMock())
        x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        y = ["A", "B"]
        with self.assertRaises(AssertionError):
            mrbbagging.fit(x, y)

    def test_fit_with_invalid_classifier(self):
        with self.assertRaises(AssertionError):
            MRBBagging(1, None)

    def test_with_invalid_k(self):
        with self.assertRaises(AssertionError):
            MRBBagging(0, tree.DecisionTreeClassifier())


if __name__ == '__main__':
    unittest.main()
