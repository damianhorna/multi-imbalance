from collections import Counter, defaultdict
from sklearn.neighbors import NearestNeighbors

import numpy as np


class SOUP(object):
    # TODO docs and tests
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        result_X, result_y = list(), list()
        neigh = NearestNeighbors(n_neighbors=self.k)
        neigh.fit(X)

        quantities = Counter(y)
        mean_quantity = np.mean(list(quantities.values()))

        for class_name, class_quantity in quantities.items():
            indices_in_class = [i for i, value in enumerate(y) if value == class_name]

            safe_levels = defaultdict(float)
            for sample_id in indices_in_class:
                neighbours_indices = neigh.kneighbors([list(X[sample_id])], return_distance=False)
                neighbours_classes = y[neighbours_indices[0]]
                neighbours_quantities = Counter(neighbours_classes)

                safe_levels[sample_id] = self._calculate_safe_level(class_name, neighbours_quantities, quantities)

            if class_quantity <= mean_quantity:
                temp_X, temp_y = self._oversample(X, y, mean_quantity, safe_levels)
            else:
                temp_X, temp_y = self._undersample(X, y, mean_quantity, safe_levels)

            result_X.extend(temp_X)
            result_y.extend(temp_y)

        X, y = result_X, result_y
        return X, y

    def _calculate_safe_level(self, class_name, neighbours_quantities, quantities):
        safe_level = 0
        for neighbour_class_name, neighbour_class_quantity in neighbours_quantities.items():
            similarity_between_classes = min(quantities[class_name], quantities[neighbour_class_name]) / max(
                quantities[class_name], quantities[neighbour_class_name])
            safe_level += neighbour_class_quantity * similarity_between_classes / self.k
        return safe_level

    def _undersample(self, X, y, goal_quantity, safe_levels):
        class_quantity = len(safe_levels)
        safe_levels_list = sorted(safe_levels.items(), key=lambda item: item[1])
        samples_to_remove_quantity = int(class_quantity - goal_quantity)
        safe_levels_list = safe_levels_list[samples_to_remove_quantity:]

        undersampled_X, undersampled_y = list(), list()
        for idx, _ in safe_levels_list:
            undersampled_X.append(X[idx])
            undersampled_y.append(y[idx])
        return undersampled_X, undersampled_y

    def _oversample(self, X, y, mean_quantity, safe_levels):
        class_quantity = len(safe_levels)
        safe_levels_list = sorted(safe_levels.items(), key=lambda item: item[1])

        oversampled_X, oversampled_y = list(), list()
        for idx, value in safe_levels_list:
            oversampled_X.append(X[idx])
            oversampled_y.append(y[idx])
        return oversampled_X, oversampled_y
