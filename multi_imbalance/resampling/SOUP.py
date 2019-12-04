import io
import pstats

from collections import Counter, defaultdict
from operator import itemgetter

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from multi_imbalance.datasets import load_datasets

import seaborn as sns

from multi_imbalance.utils.data import construct_flat_2pc_df
import matplotlib.pyplot as plt
from timeit import timeit

import cProfile


class SOUP:
    """
    Similarity Oversampling and Undersampling Preprocessing (SOUP) is an algorithm that equalizes number of samples
    in each class. It also takes care of the similarity between classes, which means that it removes samples from
    majority class, that are close to samples from the other class and duplicate samples from the minority classes,
    which are in the safest area in space
    """

    def __init__(self, k: int = 9) -> None:
        self.k = k
        self.neigh_clf = NearestNeighbors(n_neighbors=self.k + 1)
        self.quantities, self.goal_quantity = [None] * 2

    def fit_transform(self, X, y, shuffle: bool = True):
        """

        Parameters
        ----------
        X two dimensional numpy array (number of samples x number of features) with float numbers
        y one dimensional numpy array with labels for rows in X

        Returns
        -------
        Resampled X (mean class quantity * number of unique classes), y (number of rows in X) as numpy array
        """
        assert len(X.shape) == 2, 'X should have 2 dimension'
        assert X.shape[0] == y.shape[0], 'Number of labels must be equal to number of samples'

        self.quantities = Counter(y)
        self.goal_quantity = self._calculate_goal_quantity()

        dsc_maj_cls = sorted(((v, i) for v, i in self.quantities.items() if i >= self.goal_quantity), key=itemgetter(1),
                             reverse=True)
        asc_min_cls = sorted(((v, i) for v, i in self.quantities.items() if i < self.goal_quantity), key=itemgetter(1),
                             reverse=False)
        result_X, result_y = list(), list()
        for class_name, class_quantity in dsc_maj_cls:
            self.neigh_clf.fit(X)
            self._undersample(X, y, class_name, result_X, result_y)

        for class_name, class_quantity in asc_min_cls:
            self.neigh_clf.fit(X)
            self._oversample(X, y, class_name, result_X, result_y)

        if shuffle:
            result_X, result_y = sklearn.utils.shuffle(result_X, result_y)

        return np.array(result_X), np.array(result_y)

    def _construct_class_safe_levels(self, X, y, class_name) -> defaultdict:
        indices_in_class = [i for i, value in enumerate(y) if value == class_name]

        neighbour_indices = self.neigh_clf.kneighbors(X[indices_in_class], return_distance=False)
        neighbour_classes = y[neighbour_indices]

        class_safe_levels = defaultdict(float)
        for i, sample_id in enumerate(indices_in_class):
            neighbours_quantities = Counter(neighbour_classes[i])
            class_safe_levels[sample_id] = self._calculate_sample_safe_level(class_name, neighbours_quantities)

        return class_safe_levels

    def _calculate_sample_safe_level(self, class_name, neighbours_quantities: Counter):
        safe_level = 0
        q: Counter = self.quantities

        for neigh_label, neigh_q in neighbours_quantities.items():
            similarity_between_classes = min(q[class_name], q[neigh_label]) / max(q[class_name], q[neigh_label])
            safe_level += neigh_q * similarity_between_classes
        return safe_level / self.k

    def _undersample(self, X, y, class_name, result_X, result_y):
        safe_levels_of_samples_in_class = self._construct_class_safe_levels(X, y, class_name)

        class_quantity = self.quantities[class_name]
        safe_levels_list = sorted(safe_levels_of_samples_in_class.items(), key=itemgetter(1))
        samples_to_remove_quantity = max(0, int(class_quantity - self.goal_quantity))
        safe_levels_list = safe_levels_list[samples_to_remove_quantity:]

        undersampled_X = [X[idx] for idx, _ in safe_levels_list]
        undersampled_y = [y[idx] for idx, _ in safe_levels_list]

        result_X.extend(undersampled_X)
        result_y.extend(undersampled_y)

    def _oversample(self, X, y, class_name, result_X, result_y):
        safe_levels_of_samples_in_class = self._construct_class_safe_levels(X, y, class_name)
        class_quantity = self.quantities[class_name]
        safe_levels_list = sorted(safe_levels_of_samples_in_class.items(), key=itemgetter(1), reverse=True)

        oversampled_X, oversampled_y = list(), list()
        for i in range(self.goal_quantity):
            sample_level_ranking_to_duplicate: int = i % class_quantity
            sample_id, sample_safe_level = safe_levels_list[sample_level_ranking_to_duplicate]
            oversampled_X.append(X[sample_id])
            oversampled_y.append(y[sample_id])

        result_X.extend(oversampled_X)
        result_y.extend(oversampled_y)

    def _calculate_goal_quantity(self):
        max_q = max(list(self.quantities.values()))
        min_q = min(list(self.quantities.values()))
        return np.mean((min_q, max_q), dtype=int)
