from collections import Counter, defaultdict
from copy import deepcopy
from operator import itemgetter

import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors


class SOUP:
    """
    Similarity Oversampling and Undersampling Preprocessing (SOUP) is an algorithm that equalizes number of samples
    in each class. It also takes care of the similarity between classes, which means that it removes samples from
    majority class, that are close to samples from the other class and duplicate samples from the minority classes,
    which are in the safest area in space
    """

    def __init__(self, k: int = 9) -> None:
        self.k = k
        self.quantities, self.goal_quantity = [None] * 2

    def fit_transform(self, _X, _y, shuffle: bool = True):
        """

        Parameters
        ----------
        X two dimensional numpy array (number of samples x number of features) with float numbers
        y one dimensional numpy array with labels for rows in X

        Returns
        -------
        Resampled X (mean class quantity * number of unique classes), y (number of rows in X) as numpy array
        """

        X = deepcopy(_X)
        y = deepcopy(_y)

        assert len(X.shape) == 2, 'X should have 2 dimension'
        assert X.shape[0] == y.shape[0], 'Number of labels must be equal to number of samples'

        self.quantities = Counter(y)
        self.goal_quantity = self._calculate_goal_quantity()
        dsc_maj_cls = sorted(((v, i) for v, i in self.quantities.items() if i >= self.goal_quantity), key=itemgetter(1),
                             reverse=True)
        asc_min_cls = sorted(((v, i) for v, i in self.quantities.items() if i < self.goal_quantity), key=itemgetter(1),
                             reverse=False)

        for class_name, class_quantity in dsc_maj_cls:
            X, y = self._undersample(X, y, class_name)

        for class_name, class_quantity in asc_min_cls:
            X, y = self._oversample(X, y, class_name)

        if shuffle:
            X, y = sklearn.utils.shuffle(X, y)

        return np.array(X), np.array(y)

    def _construct_class_safe_levels(self, X, y, class_name) -> defaultdict:
        indices_in_class = [i for i, value in enumerate(y) if value == class_name]

        neigh_clf = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        neighbour_indices = neigh_clf.kneighbors(X[indices_in_class], return_distance=False)[:, 1:]
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
            safe_level += neigh_q * similarity_between_classes / self.k

        if safe_level > 1:
            raise ValueError(f'Safe level is bigger than 1: {safe_level}')

        return safe_level

    def _undersample(self, X, y, class_name):
        safe_levels_of_samples_in_class = self._construct_class_safe_levels(X, y, class_name)

        class_quantity = self.quantities[class_name]
        safe_levels_list = sorted(safe_levels_of_samples_in_class.items(), key=itemgetter(1))
        samples_to_remove_quantity = max(0, int(class_quantity - self.goal_quantity))
        if samples_to_remove_quantity > 0:
            remove_indices = list(map(itemgetter(0), safe_levels_list[:samples_to_remove_quantity]))
            X = np.delete(X, remove_indices, axis=0)
            y = np.delete(y, remove_indices, axis=0)

        return X, y

    def _oversample(self, X, y, class_name):
        safe_levels_of_samples_in_class = self._construct_class_safe_levels(X, y, class_name)
        class_quantity = self.quantities[class_name]
        safe_levels_list = list(sorted(safe_levels_of_samples_in_class.items(), key=itemgetter(1), reverse=True))

        difference = self.goal_quantity - class_quantity
        while difference > 0:
            quantity_items_to_copy = min(difference, class_quantity)
            indices_to_copy = list(map(itemgetter(0), safe_levels_list[:quantity_items_to_copy]))
            X = np.vstack((X, X[indices_to_copy]))
            y = np.hstack((y, y[indices_to_copy]))
            difference -= quantity_items_to_copy

        return X, y

    def _calculate_goal_quantity(self):
        max_q = max(list(self.quantities.values()))
        min_q = min(list(self.quantities.values()))
        return np.mean((min_q, max_q), dtype=int)
