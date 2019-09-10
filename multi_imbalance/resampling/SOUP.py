from collections import Counter, defaultdict

import sklearn
from sklearn.neighbors import NearestNeighbors
from nptyping import Array

import numpy as np


class SOUP(object):
    """
    Similarity Oversampling and Undersampling Preprocessing (SOUP) is an algorithm that equalizes number of samples
    in each class. It also takes care of the similarity between classes, which means that it removes samples from
    majority class, that are close to samples from the other class and duplicate samples from the minority classes,
    which are in the safest area in space
    """

    def __init__(self, k: int = 9) -> None:
        self.k = k
        self.neigh_clf = NearestNeighbors(n_neighbors=self.k)
        self.quantities, self.goal_quantity = [None] * 2

    def fit_transform(self, X: Array[float], y: Array, shuffle: bool = True) -> (Array[float], Array):
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

        result_X, result_y = list(), list()
        self.neigh_clf.fit(X)
        self.quantities = Counter(y)
        self.goal_quantity = np.mean(list(self.quantities.values()), dtype=int)

        for class_name, class_quantity in self.quantities.items():

            class_safe_levels: defaultdict = self._construct_class_safe_levels(X, y, class_name)

            if class_quantity <= self.goal_quantity:
                temp_X, temp_y = self._oversample(X, y, class_safe_levels)
            else:
                temp_X, temp_y = self._undersample(X, y, class_safe_levels)

            result_X.extend(temp_X)
            result_y.extend(temp_y)

        if shuffle:
            result_X, result_y = sklearn.utils.shuffle(result_X, result_y)

        return np.array(result_X), np.array(result_y)

    def _construct_class_safe_levels(self, X: Array[float], y: Array, class_name) -> defaultdict:
        indices_in_class = [i for i, value in enumerate(y) if value == class_name]

        class_safe_levels = defaultdict(float)
        for sample_id in indices_in_class:
            neighbours_quantities = self._calculate_neighbour_quantities_for_sample(X, y, sample_id)
            class_safe_levels[sample_id] = self._calculate_sample_safe_level(class_name, neighbours_quantities)
        return class_safe_levels

    def _calculate_neighbour_quantities_for_sample(self, X: Array[float], y: Array, sample_id):
        sample_row = [list(X[sample_id])]
        neighbours_indices = self.neigh_clf.kneighbors(sample_row, return_distance=False)[0]
        neighbours_classes = y[neighbours_indices]
        neighbours_quantities = Counter(neighbours_classes)
        return neighbours_quantities

    def _calculate_sample_safe_level(self, class_name, neighbours_quantities: Counter):
        safe_level = 0
        q: Counter = self.quantities

        for neighbour_class_name, neighbour_class_quantity in neighbours_quantities.items():
            similarity_between_classes = min(q[class_name], q[neighbour_class_name]) / max(q[class_name],
                                                                                           q[neighbour_class_name])
            safe_level += neighbour_class_quantity * similarity_between_classes / self.k
        return safe_level

    def _undersample(self, X: Array[float], y: Array, safe_levels_of_samples_in_class: defaultdict) -> (
            Array[float], Array):
        if len(safe_levels_of_samples_in_class) < self.goal_quantity:
            raise AttributeError(
                "Quantity of classes safe_levels should be higher than goal quantity for undersampling")

        class_quantity = len(safe_levels_of_samples_in_class)
        safe_levels_list = sorted(safe_levels_of_samples_in_class.items(), key=lambda item: item[1])
        samples_to_remove_quantity = int(class_quantity - self.goal_quantity)
        safe_levels_list = safe_levels_list[samples_to_remove_quantity:]

        undersampled_X = [X[idx] for idx, _ in safe_levels_list]
        undersampled_y = [y[idx] for idx, _ in safe_levels_list]

        return undersampled_X, undersampled_y

    def _oversample(self, X: Array[float], y: Array, safe_levels_of_samples_in_class: defaultdict) -> (
            Array[float], Array):
        if len(safe_levels_of_samples_in_class) > self.goal_quantity:
            raise AttributeError("Quantity of classes safe_levels should be lower than goal quantity for oversampling")

        class_quantity = len(safe_levels_of_samples_in_class)
        safe_levels_list = sorted(safe_levels_of_samples_in_class.items(), key=lambda item: item[1], reverse=True)

        oversampled_X, oversampled_y = list(), list()
        for i in range(self.goal_quantity):
            sample_level_ranking_to_duplicate: int = i % class_quantity
            sample_id, sample_safe_level = safe_levels_list[sample_level_ranking_to_duplicate]
            oversampled_X.append(X[sample_id])
            oversampled_y.append(y[sample_id])

        return oversampled_X, oversampled_y
