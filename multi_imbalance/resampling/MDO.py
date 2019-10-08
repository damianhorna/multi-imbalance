from random import sample
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import numpy as np


class MDO(object):
    """
    Mahalanbois Distance Oversampling is an algorithm that oversamples all classes to a quantity of the major class.
    Samples for oversampling are chosen based on their k neighbours and new samples are created in random place but
    with the same Mahalanbois distance from the centre of class to chosen sample.

    """

    def __init__(self, k=9, k1_frac=.5):
        self.knn = NearestNeighbors(n_neighbors=k)
        self.k2 = k
        self.k1 = int(k * k1_frac)

    def fit_transform(self, X, y):
        """

        Parameters
        ----------
        X two dimensional numpy array (number of samples x number of features) with float numbers
        y one dimensional numpy array with labels for rows in X

        Returns
        -------
        Resampled X and y
        """
        self.knn.fit(X)
        oversampled_X, oversampled_y = X.copy(), y.copy()
        quantities = Counter(y)
        goal_quantity = int(max(list(quantities.values())))

        labels = list(set(y))
        for class_label in labels:
            SC_minor, weights = self._choose_samples(X, y, class_label)
            if (len(SC_minor)) == 0:
                continue

            u = np.mean(SC_minor, axis=0)
            Z = SC_minor - u
            n_components = min(Z.shape)
            pca = PCA(n_components=n_components).fit(Z)
            T = pca.transform(Z)
            V = np.var(T, axis=0)
            oversampling_rate = goal_quantity - quantities[class_label]

            if oversampling_rate > 0:
                S_temp = self._MDO_oversampling(T, V, oversampling_rate, weights)
                S_temp = pca.inverse_transform(S_temp) + u

                oversampled_X = np.vstack((oversampled_X, S_temp))
                oversampled_y = np.hstack((oversampled_y, np.array([class_label] * oversampling_rate)))

        return oversampled_X, oversampled_y

    def _choose_samples(self, X, y, class_label):
        S_minor_class_indices = [i for i, value in enumerate(y) if value == class_label]

        S_minor = X[S_minor_class_indices]
        class_label = y[S_minor_class_indices[0]]

        minority_class_neighbours_indices = self.knn.kneighbors(S_minor, return_distance=False)

        quantity_with_same_label_in_neighbourhood = list()
        for i in range(len(S_minor)):
            sample_neighbours_indices = minority_class_neighbours_indices[i][1:]
            quantity_sample_neighbours_indices_with_same_label = sum(y[sample_neighbours_indices] == class_label)
            quantity_with_same_label_in_neighbourhood.append(quantity_sample_neighbours_indices_with_same_label)
        num = np.array(quantity_with_same_label_in_neighbourhood)

        SC_minor = S_minor[num >= self.k1]

        weights = num[num >= self.k1] / self.k2
        weights[weights == 0] = 1
        weights = weights / np.sum(weights)

        return SC_minor, weights

    @staticmethod
    def _MDO_oversampling(T, V, oversampling_rate, weights):
        S_temp = list()
        for _ in range(oversampling_rate):
            idx = np.random.choice(np.arange(len(T)), p=weights)
            X = np.square(T[idx])
            a = np.sum(X / V)
            alpha_V = a * V

            s = 0
            features_vector = list()
            for alpha_V_j in alpha_V[:-1]:
                sqrt_avj = np.sqrt(alpha_V_j)
                r = np.random.uniform(low=-sqrt_avj, high=sqrt_avj)
                s += r ** 2 / sqrt_avj
                features_vector.append(r)

            last = (1 - s) * alpha_V[-1]
            last_feature = np.sqrt(last) if last > 0 else 0
            random_last_feature = sample([-last_feature, last_feature], 1)[0]

            features_vector.append(random_last_feature)
            S_temp.append(features_vector)

        return np.array(S_temp)
