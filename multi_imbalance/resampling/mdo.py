from collections import Counter
from typing import Dict, List, Tuple, Union

import numpy as np
from imblearn.base import BaseSampler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from multi_imbalance.utils.data import construct_maj_int_min


class MDO(BaseSampler):
    """
    Mahalanbois Distance Oversampling is an algorithm that oversamples all classes to a quantity of the major class.
    Samples for oversampling are chosen based on their k neighbours and new samples are created in random place but
    with the same Mahalanbois distance from the centre of class to chosen sample.

    """

    def __init__(
        self,
        k: int = 5,
        k1_frac: float = 0.4,
        seed: int = 0,
        prop: int = 1,
        maj_int_min: Union[Dict[str, List[int]], None] = None,
    ) -> None:
        """
        :param k:
            Number of neighbours considered during the neighbourhood analysis
        :param k1_frac:
            Ratio of the number of neighbours in the sample class to all neighbours in the neighbourhood.
            If the ratio is greater, the example will not be considered noise
        :param seed:
        :param prop:
            Oversampling ratio, if equal to one the class size after resampling will be equal to the size of
            the largest class
        :param maj_int_min:
            dict {'maj': majority class labels, 'min': minority class labels}
        """
        super().__init__()
        self._sampling_type = "over-sampling"
        self.knn = NearestNeighbors(n_neighbors=k)
        self.k2 = k
        self.k1 = int(k * k1_frac)
        self.random_state = check_random_state(seed)
        self.X, self.y = None, None
        self.prop = prop
        self.class_balances = maj_int_min

    def _fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X:
            two dimensional numpy array (number of samples x number of features) with float numbers
        :param y:
            one dimensional numpy array with labels for rows in X
        :return:
            resampled X, resampled y
        """
        if self.class_balances is None:
            self.class_balances = construct_maj_int_min(y)

        self.knn.fit(X)
        self.X, self.y = X, y

        oversampled_X, oversampled_y = self.X.copy(), self.y.copy()
        quantities = Counter(self.y)
        goal_quantity = int(max(list(quantities.values())))
        labels = list(set(self.y))
        minority_classes = self.class_balances["min"]

        for class_label in labels:
            if minority_classes is not None and class_label not in minority_classes:
                continue

            chosen_minor_class_samples_to_oversample, weights = self._choose_samples(class_label)
            if len(chosen_minor_class_samples_to_oversample) == 0:
                continue

            oversampling_rate = int((goal_quantity - quantities[class_label]) * self.prop)
            if oversampling_rate > 0:
                if len(chosen_minor_class_samples_to_oversample) == 1:
                    oversampled_set = np.repeat(
                        chosen_minor_class_samples_to_oversample,
                        oversampling_rate,
                        axis=0,
                    )
                else:
                    chosen_samples_features_mean = np.mean(chosen_minor_class_samples_to_oversample, axis=0)
                    zero_mean_samples = chosen_minor_class_samples_to_oversample - chosen_samples_features_mean

                    n_components = min(zero_mean_samples.shape)
                    pca = PCA(n_components=n_components).fit(zero_mean_samples)

                    uncorrelated_samples = pca.transform(zero_mean_samples)
                    variables_variance = np.diag(np.cov(uncorrelated_samples, rowvar=False))

                    oversampled_set = self._MDO_oversampling(
                        uncorrelated_samples,
                        variables_variance,
                        oversampling_rate,
                        weights,
                    )
                    oversampled_set = pca.inverse_transform(oversampled_set) + chosen_samples_features_mean

                oversampled_X = np.vstack((oversampled_X, oversampled_set))
                oversampled_y = np.hstack((oversampled_y, np.array([class_label] * oversampling_rate)))

        return oversampled_X, oversampled_y

    def _choose_samples(self, class_label: str) -> Tuple[np.ndarray, np.ndarray]:
        minor_class_indices = [i for i, value in enumerate(self.y) if value == class_label]
        minor_set = self.X[minor_class_indices]

        quantity_same_class_neighbours = self.calculate_same_class_neighbour_quantities(minor_set, class_label)
        chosen_minor_class_samples_to_oversample = minor_set[quantity_same_class_neighbours >= self.k1]

        weights = quantity_same_class_neighbours[quantity_same_class_neighbours >= self.k1] / self.k2
        weights_sum = np.sum(weights)

        if weights_sum != 0:
            weights /= np.sum(weights)
        elif len(weights) > 0:
            value = 1 / len(weights)
            weights += value

        return chosen_minor_class_samples_to_oversample, weights

    def _MDO_oversampling(self, T: np.ndarray, v: np.ndarray, oversampling_rate: int, weights: np.ndarray) -> np.ndarray:
        oversampled_set = list()
        V = np.clip(np.copy(v), a_min=0.001, a_max=None)
        for _ in range(oversampling_rate):
            idx = self.random_state.choice(np.arange(len(T)), p=weights)
            X = np.square(T[idx])
            a = np.sum(X / V)
            alpha_V = a * V
            alpha_V[alpha_V < 0.001] = 0.001

            s = 0
            features_vector = list()
            for alpha_V_j in alpha_V[:-1]:
                sqrt_avj = np.sqrt(alpha_V_j)
                r = self.random_state.uniform(low=-sqrt_avj, high=sqrt_avj)
                s += r**2 / alpha_V_j
                features_vector.append(r)

            last = (1 - s) * alpha_V[-1]
            last_feature = np.sqrt(last) if last > 0 else 0
            random_last_feature = self.random_state.choice([-last_feature, last_feature], 1)[0]

            features_vector.append(random_last_feature)
            oversampled_set.append(features_vector)

        return np.array(oversampled_set)

    def calculate_same_class_neighbour_quantities(self, S_minor: np.ndarray, S_minor_label: str) -> np.ndarray:
        minority_class_neighbours_indices = self.knn.kneighbors(S_minor, return_distance=False)
        quantity_with_same_label_in_neighbourhood = list()
        for i in range(len(S_minor)):
            sample_neighbours_indices = minority_class_neighbours_indices[i][1:]
            quantity_sample_neighbours_indices_with_same_label = sum(self.y[sample_neighbours_indices] == S_minor_label)
            quantity_with_same_label_in_neighbourhood.append(quantity_sample_neighbours_indices_with_same_label)
        return np.array(quantity_with_same_label_in_neighbourhood)
