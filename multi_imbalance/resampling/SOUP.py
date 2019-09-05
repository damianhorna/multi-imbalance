from collections import Counter, defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle

import numpy as np


class SOUP(object):
    # TODO docs and tests
    def __init__(self, k=5):
        self.k = k
        self.neigh_clf = NearestNeighbors(n_neighbors=self.k)
        self.quantities = None

    def fit(self, X, y):
        result_X, result_y = list(), list()
        self.neigh_clf.fit(X)
        self.quantities = Counter(y)
        mean_quantity = np.mean(list(self.quantities.values()), dtype=int)

        for class_name, class_quantity in self.quantities.items():
            class_safe_levels = self._construct_class_safe_levels(X, y, class_name)

            if class_quantity <= mean_quantity:
                temp_X, temp_y = self._oversample(X, y, mean_quantity, class_safe_levels)
            else:
                temp_X, temp_y = self._undersample(X, y, mean_quantity, class_safe_levels)

            result_X.extend(temp_X)
            result_y.extend(temp_y)

        shuffle(np.array(result_X), np.array(result_y))
        return result_X, result_y

    def _construct_class_safe_levels(self, X, y, class_name):
        indices_in_class = [i for i, value in enumerate(y) if value == class_name]

        class_safe_levels = defaultdict(float)
        for sample_id in indices_in_class:
            neighbours_indices = self.neigh_clf.kneighbors([list(X[sample_id])], return_distance=False)
            neighbours_classes = y[neighbours_indices[0]]
            neighbours_quantities = Counter(neighbours_classes)
            class_safe_levels[sample_id] = self._calculate_sample_safe_level(class_name, neighbours_quantities)

        return class_safe_levels

    def _calculate_sample_safe_level(self, class_name, neighbours_quantities):
        safe_level = 0
        q = self.quantities

        for neighbour_class_name, neighbour_class_quantity in neighbours_quantities.items():
            similarity_between_classes = min(q[class_name], q[neighbour_class_name]) / max(q[class_name],
                                                                                           q[neighbour_class_name])
            safe_level += neighbour_class_quantity * similarity_between_classes / self.k
        return safe_level

    @staticmethod
    def _undersample(X, y, goal_quantity, class_safe_levels):
        class_quantity = len(class_safe_levels)
        safe_levels_list = sorted(class_safe_levels.items(), key=lambda item: item[1])
        samples_to_remove_quantity = int(class_quantity - goal_quantity)
        safe_levels_list = safe_levels_list[samples_to_remove_quantity:]

        undersampled_X, undersampled_y = list(), list()
        for idx, _ in safe_levels_list:
            undersampled_X.append(X[idx])
            undersampled_y.append(y[idx])
        return undersampled_X, undersampled_y

    @staticmethod
    def _oversample(X, y, mean_quantity, class_safe_levels):
        class_quantity = len(class_safe_levels)
        safe_levels_list = sorted(class_safe_levels.items(), key=lambda item: item[1], reverse=True)

        oversampled_X, oversampled_y = list(), list()
        for i in range(mean_quantity):
            sample_level_ranking_to_duplicate = i % class_quantity
            sample_id, sample_safe_level = safe_levels_list[sample_level_ranking_to_duplicate]
            oversampled_X.append(X[sample_id])
            oversampled_y.append(y[sample_id])

        return oversampled_X, oversampled_y
