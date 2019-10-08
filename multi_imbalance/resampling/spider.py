import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class SPIDER3:
    def __init__(self, k, cost, majority_classes, intermediate_classes, minority_classes):
        self.k = k
        self.neigh_clf = NearestNeighbors(n_neighbors=self.k)
        self.cost = cost
        self.majority_classes = majority_classes
        self.intermediate_classes = intermediate_classes
        self.minority_classes = minority_classes
        self.AS, self.RS = np.array([]), np.array([])  # RS - examples from majority class that can be relabeled


    def fit_transform(self, X, y):
        self.DS = np.append(X, y.reshape(y.shape[0], 1), axis=1)

        for clazz in self.majority_classes:
            DS_clazz = self.DS[self.DS[:, -1] == clazz]
            for x in DS_clazz:
                if clazz not in self._min_cost_classes(x, self.DS):
                    self._union(self.RS, np.array([x]))

        self.DS = self._setdiff(self.DS, self.RS)

        for clazz in self.intermediate_classes + self.minority_classes:
            DS_clazz = self.DS[self.DS[:, -1] == clazz]
            for x in DS_clazz:
                self.relabel_nn(x)

            if self.AS.size != 0:
                AS_clazz = self.AS[self.AS[:, -1] == clazz]
            else:
                AS_clazz = np.array([])

            for x in self._union(DS_clazz, AS_clazz):
                self.clean_nn(x)

            for x in DS_clazz:
                self.amplify(x)

        self.DS = self._union(self.DS, self.AS)

        return self.DS[:, :-1], self.DS[:, -1]

    def _min_cost_classes(self, x, DS):
        C = self.majority_classes + self.intermediate_classes + self.minority_classes
        vals = []
        for cj in C:
            s = 0
            for ci in C:  # if ci != cj?
                s += (self._knn(x, DS, ci).shape[0] / self.k) * self.cost[C.index(ci), C.index(cj)]
            vals.append(s)
        C = np.array(C)
        vals = np.array(vals)
        return C[vals == vals[
            np.argmin(vals)]]  # any arg that minimizes or all of them where for example there are two with same value?

    def _setdiff(self, S1, S2):
        for element in S2.tolist():
            if element in S1.tolist():
                S1 = np.delete(S1, S1.tolist().index(element), 0)
        return S1

    def _union(self, arr1, arr2):
        if arr1.size == 0:
            return arr2
        elif arr2.size == 0:
            return arr1
        else:
            result = arr1.copy()
            for x2 in arr2:
                elem_uniq = True
                for x1 in arr1:
                    if all(x1 == x2):
                        elem_uniq = False
                        break
                if elem_uniq:
                    result = np.append(arr1, np.array([x2]), axis=0)
            return result

    def _intersect(self, arr1, arr2):
        if arr1.size == 0 or arr2.size == 0:
            return np.array([])

        result = np.array([])
        for x1 in arr1:
            for x2 in arr2:
                if all(x1 == x2):
                    result = self._union(result, np.array([x1]))
        return result

    def relabel_nn(self, x):
        neighbourhood_candidates = self._knn(x, self._union(self.DS, self._union(self.AS, self.RS))) # do we actually need the union with DS and AS? We're taking intersect with RS in the next step anyway
        TS = self._intersect(self.RS, neighbourhood_candidates)  # TS - neighbours from majority class that can be relabeled
        while TS.shape[0] > 0 and \
                any(majority_class in self._min_cost_classes(x, self._union(self.DS, self._union(self.AS, self.RS)))
                    for majority_class in self.majority_classes):
            y = self.nearest(x, TS)
            TS = self._setdiff(TS, np.array([y]))
            self.RS = self._setdiff(self.RS, np.array([y]))
            y[-1] = x[-1]
            self.AS = self._union(self.AS, np.array([y]))

    def nearest(self, x, TS):
        clf = NearestNeighbors(n_neighbors=1).fit(TS[:, :-1])
        indices = clf.kneighbors([x[:-1]], return_distance=False)
        return TS[indices[0]][0]

    def clean_nn(self, x):
        for majority_class in self.majority_classes:
            TS = self._knn(x, self._union(self.DS, self._union(self.AS, self.RS)), majority_class)
            while TS.shape[0] > 0 and \
                    any(majority_class in self._min_cost_classes(x, self._union(self.DS, self._union(self.AS, self.RS)))
                        for majority_class in self.majority_classes):
                y = self.nearest(x, TS)
                TS = self._setdiff(TS, np.array([y]))
                self.DS = self._setdiff(self.DS, np.array([y]))
                self.RS = self._setdiff(self.RS, np.array([y]))

    def _knn(self, x, DS, c=None):
        self.neigh_clf.fit(DS[:, :-1])
        indices = self.neigh_clf.kneighbors([x[:-1]], return_distance=False)[0]
        if c is not None:
            result = []
            for idx in indices:
                if class_of(DS[idx]) in c:
                    result.append(DS[idx])
            return np.array(result)
        else:
            return DS[indices]

    def amplify(self, x):
        while class_of(x) not in self._min_cost_classes(x, self._union(self.DS, self._union(self.AS, self.RS))):
            y = x.copy()
            self.AS = self._union(self.AS, np.asarray([y]))


def class_of(example):
    return example[-1]


def plot_multi_dimensional_data(X, y, ax=None):
    """
    This function reduce quantity of dimensions to 2 principal components and prepare pretty scatter plot for your data
    Parameters
    ----------
    X multi dimensional numpy array (at least 2 dimensions)
    y one dimensional numpy array with labels
    ax optional parameter for subplots
    Returns None
    -------
    """

    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

    y = pd.DataFrame({'y': y})

    X_df = pd.DataFrame(data=X, columns=['x1', 'x2'])
    df = pd.concat([X_df, y], axis=1)

    sns.scatterplot(x='x1', y='x2', hue='y', style='y', data=df, alpha=0.7, ax=ax, legend=False)


if __name__ == "__main__":
    rc = {'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'}
    sns.set_style('darkgrid', rc=rc)

    # TODO replace it by correct file in repository
    ecoli_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'
    df = pd.read_csv(ecoli_url, delim_whitespace=True, header=None,
                     names=['name', '1', '2', '3', '4', '5', '6', '7', 'class'])

    X, y = df.iloc[:, 1:8].to_numpy(), df['class'].to_numpy()
    print(X[:5])
    print(y[:5])
    cost = np.random.rand(64).reshape((8, 8))  # np.ones((8, 8))
    for i in range(8):
        cost[i][i] = 0

    clf = SPIDER3(k=3, cost=cost, majority_classes=['cp', 'im'],
                  intermediate_classes=['pp', 'imU', 'om'], minority_classes=['imS', 'imL', 'omL'])
    transformed = clf.fit_transform(X.astype(np.float64), y)
    print("Done")
