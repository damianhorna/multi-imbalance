import pytest
from sklearn.neighbors import KNeighborsClassifier

from multi_imbalance.ensemble.soup_bagging import SOUPBagging
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

y_test = np.array([0, 1, 0, 0])


def test_soubagging():
    clf = KNeighborsClassifier()
    maj_int_min = {'maj': [0], 'int': [], 'min': [1]}
    clf = SOUPBagging(clf, n_classifiers=2, maj_int_min=maj_int_min)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    assert all(y_pred == y_test)
    y_pred = clf.predict(X_test, strategy='mixed', maj_int_min=maj_int_min)
    assert all(y_pred == y_test)
    y_pred = clf.predict(X_test, strategy='optimistic')
    assert all(y_pred == y_test)
    y_pred = clf.predict(X_test, strategy='pessimistic')
    assert all(y_pred == y_test)
    y_pred = clf.predict(X_test, strategy='global')
    assert all(y_pred == y_test)


def test_exception():
    clf = KNeighborsClassifier()
    maj_int_min = {'maj': [0], 'int': [], 'min': [1]}
    clf = SOUPBagging(clf, n_classifiers=2, maj_int_min=maj_int_min)
    clf.fit(X_train, y_train)
    with pytest.raises(KeyError):
        clf.predict(X_test, 'incorrect')


def test_default_classifier():
    maj_int_min = {'maj': [0], 'int': [], 'min': [1]}
    clf = SOUPBagging(n_classifiers=2, maj_int_min=maj_int_min)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    assert all(y_pred == y_test)
