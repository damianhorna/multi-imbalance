from collections import Counter, defaultdict

import numpy as np
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler, RobustScaler

from multi_imbalance.resampling.SOUP import SOUP

X = np.array([
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
    [0.9287003, 0.97580299],
    [0.9584236, 0.10536541],
    [0.96983103, 0.87666093],
    [0.97352367, 0.78807909],
])
from multi_imbalance.utils.plot import plot_multi_dimensional_data

y_balanced = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
y_balanced_quantities = Counter({0: 10, 1: 10})
y_balanced_0_class_safe_levels = defaultdict(float,
                                             {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0,
                                              9: 1.0})
y_balanced_1_class_safe_levels = defaultdict(float,
                                             {10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0,
                                              18: 1.0, 19: 1.0})

y_imbalanced_easy = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])
y_imbalanced_easy_quantities = Counter({0: 14, 1: 6})
y_imbalanced_easy_0_class_safe_levels = defaultdict(float,
                                                    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
                                                     8: 1.0, 9: 1.0, 10: 0.8857142857142858, 11: 0.7714285714285714,
                                                     12: 0.7714285714285714, 17: 0.7714285714285714})

y_imbalanced_easy_1_class_safe_levels = defaultdict(float, {13: 0.6571428571428571, 14: 0.7714285714285714,
                                                            15: 0.8857142857142858, 16: 0.8857142857142858,
                                                            18: 0.8857142857142858, 19: 0.8857142857142858})

y_imbalanced_hard = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0])
y_imbalanced_hard_quantities = Counter({0: 14, 1: 6})
y_imbalanced_hard_quantities_0_class_safe_levels = defaultdict(float, {0: 0.8857142857142858, 1: 0.8857142857142858,
                                                                       2: 0.8857142857142858, 3: 0.8857142857142858,
                                                                       4: 0.8857142857142858, 5: 0.7714285714285714,
                                                                       7: 0.7714285714285714, 10: 0.7714285714285714,
                                                                       11: 0.7714285714285714, 12: 0.7714285714285714,
                                                                       13: 0.7714285714285714, 15: 0.7714285714285714,
                                                                       17: 0.7714285714285714, 19: 0.7714285714285714})
y_imbalanced_hard_quantities_1_class_safe_levels = defaultdict(float, {6: 0.6571428571428571, 8: 0.6571428571428571,
                                                                       9: 0.6571428571428571, 14: 0.5428571428571429,
                                                                       16: 0.6571428571428571, 18: 0.6571428571428571})

import seaborn as sns

sns.set_style('darkgrid')

plot_multi_dimensional_data(X, y_imbalanced_hard)
y = np.array([])

import matplotlib.pyplot as plt

plt.show()

clf = SOUP(k=5)
clf.fit_transform(X, y_balanced)


def test_oversampling_when_typical_situation():
    pass
