from random import sample
from collections import Counter

from multi_imbalance.utils.data import construct_flat_2pc_df
from sklearn.neighbors import NearestNeighbors
import numpy as np


class MDO(object):
    def __init__(self, k=9, k1_frac=.5):
        self.nn = NearestNeighbors(n_neighbors=k)
        self.k2 = k
        self.k1 = int(k * k1_frac)

    def fit_transform(self, X, y):
        self.nn.fit(X)
        oversampled_X, oversampled_y = X.copy(), y.copy()
        quantities = Counter(y)
        goal_quantity = int(max(list(quantities.values())))

        labels = list(set(y))
        for class_label in labels:
            S_minor_class_indices = [i for i, value in enumerate(y) if value == class_label]
            SC_minor, weights = self._choose_samples(X, y, S_minor_class_indices)
            if (len(SC_minor)) == 0:
                # TODO?
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

    def _choose_samples(self, X, y, S_minor_class_indices):
        S_minor = X[S_minor_class_indices]
        class_label = y[S_minor_class_indices[0]]

        minority_class_neighbours_indices = self.nn.kneighbors(S_minor, return_distance=False)

        quantity_with_same_label_in_neighbourhood = list()
        for i in range(len(S_minor)):
            sample_neighbours_indices = minority_class_neighbours_indices[i][1:]
            quantity_sample_neighbours_indices_with_same_label = sum(y[sample_neighbours_indices] == class_label)
            quantity_with_same_label_in_neighbourhood.append(quantity_sample_neighbours_indices_with_same_label)
        num = np.array(quantity_with_same_label_in_neighbourhood)

        SC_minor = S_minor[num >= self.k1]

        weights = num[num >= self.k1] / self.k2
        weights[weights == 0] = 1  # TODO?
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

            # TODO >1?
            last = (1 - s) * alpha_V[-1]
            last_feature = np.sqrt(last) if last > 0 else 0
            random_last_feature = sample([-last_feature, last_feature], 1)[0]

            features_vector.append(random_last_feature)
            S_temp.append(features_vector)

        return np.array(S_temp)


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# # TODO replace it by correct file in repository
# ecoli_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'
# df = pd.read_csv(ecoli_url, delim_whitespace=True, header=None,
#                  names=['name', '1', '2', '3', '4', '5', '6', '7', 'class'])
#
# X, y = df.iloc[:, 1:8].to_numpy(), df['class'].to_numpy()
# print(X[:5])
# print(y[:5])
#
# clf = MDO(k1_frac=0)
# resampled_X, resampled_y = clf.fit_transform(X, y)
# print(resampled_X)
#
# pca = PCA(n_components=2)
# pca.fit(X)
#
# fig, axs = plt.subplots(ncols=2, nrows=2)
# fig.set_size_inches(16, 10)
# axs = axs.flatten()
#
# sns.countplot(y, ax=axs[0])
# X = pca.transform(X)
# df = construct_flat_2pc_df(X, y)
# sns.scatterplot(x='x1', y='x2', hue='y', style='y', data=df, alpha=0.7, ax=axs[1], legend=False)
#
# sns.countplot(resampled_y, ax=axs[2])
# resampled_X = pca.transform(resampled_X)
# df = construct_flat_2pc_df(resampled_X, resampled_y)
# sns.scatterplot(x='x1', y='x2', hue='y', style='y', data=df, alpha=0.7, ax=axs[3], legend=False)
# plt.show()
mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diag

import matplotlib.pyplot as plt
x, y = np.random.multivariate_normal(mean, cov, 10).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()