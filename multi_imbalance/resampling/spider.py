import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class SPIDER3:
    """
    SPIDER3 algorithm implementation for selective preprocessing of multi-class imbalanced data sets.

    Reference:
    Wojciechowski, S., Wilk, S., Stefanowski, J.: An Algorithm for Selective Preprocessing
    of Multi-class Imbalanced Data. Proceedings of the 10th International Conference
    on Computer Recognition Systems CORES 2017

    Methods
    ----------
    fit_transform(X, y)
        Performs resampling of X.
    """

    def __init__(self, k, cost, majority_classes, intermediate_classes, minority_classes):
        """
        Parameters
        ----------
        :param k:
            Number of nearest neighbors considered while resampling.
        :param cost:
            The cost matrix. An element c[i, j] of this matrix represents the cost
            associated with misclassifying an example from class i as class one from class j.
        :param majority_classes:
            List of majority classes.
        :param intermediate_classes:
            List of intermediate classes.
        :param minority_classes:
            List of minority classes.
        """

        self.k = k
        self.neigh_clf = NearestNeighbors(n_neighbors=self.k)
        self.cost = cost
        self.majority_classes = majority_classes
        self.intermediate_classes = intermediate_classes
        self.minority_classes = minority_classes
        self.AS, self.RS = np.array([]), np.array([])  # RS - examples from majority class that can be relabeled

    def fit_transform(self, X, y):
        """

        :param X:
            Numpy array of examples that is the subject of resampling.
        :param y:
            Numpy array of labels corresponding to examples from X.
        :return:
            Resampled X along with accordingly modified labels.
        """

        self.DS = np.append(X, y.reshape(y.shape[0], 1), axis=1)
        self.calculate_weak_majority_examples()
        self.DS = self._setdiff(self.DS, self.RS)

        for clazz in self.intermediate_classes + self.minority_classes:
            DS_clazz = self.DS[self.DS[:, -1] == clazz]
            for x in DS_clazz:
                self._relabel_nn(x)

            if self.AS.size != 0:
                AS_clazz = self.AS[self.AS[:, -1] == clazz]
            else:
                AS_clazz = np.array([])

            for x in self._union(DS_clazz, AS_clazz):
                self._clean_nn(x)

            for x in DS_clazz:
                self._amplify(x)

        self.DS = self._union(self.DS, self.AS)

        return self.DS[:, :-1], self.DS[:, -1]

    def calculate_weak_majority_examples(self):
        """
        Calculates weak majority examples and appends them to the RS set.
        :return:
        """

        for majority_class in self.majority_classes:
            majority_examples = self.DS[self.DS[:, -1] == majority_class]
            for x in majority_examples:
                if majority_class not in self._min_cost_classes(x, self.DS):
                    self.RS = self._union(self.RS, np.array([x]))

    def _min_cost_classes(self, x, DS):
        """
        Utility function that aims to identify minimum-cost classes, i.e. classes leading
        to the minimum cost after being (mis)classified as classes appearing in the neighborhood of x.

        :param x:
            Single observation
        :param DS:
            DS
        :return:
            List of classes associated with minimal cost of misclassification.
        """

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
            np.argmin(vals)]]

    @staticmethod
    def _setdiff(arr1, arr2):
        """
        Performs the difference over two numpy arrays.

        :param arr1:
            Numpy array number 1.
        :param arr2:
            Numpy array number 2.
        :return:
            Result of the difference of arr1 and arr2.
        """

        for element in arr2.tolist():
            if element in arr1.tolist():
                arr1 = np.delete(arr1, arr1.tolist().index(element), 0)
        return arr1

    @staticmethod
    def _union(arr1, arr2):
        """
        Performs the union over two numpy arrays
        (not removing duplicates, as it's how the algorithm SPIDER3 actually works).

        :param arr1:
            Numpy array number 1.
        :param arr2:
            Numpy array number 2.
        :return:
            The union of arr1 and arr2.
        """

        if arr1.size == 0:
            return arr2
        elif arr2.size == 0:
            return arr1
        else:
            return np.append(arr1, arr2, axis=0)

    def _intersect(self, arr1, arr2):
        """
        Performs the intersection operation over two numpy arrays (not removing duplicates).

        :param arr1:
            Numpy array number 1.
        :param arr2:
            Numpy array number 2.
        :return:
            The intersection of arr1 and arr2.
        """

        if arr1.size == 0 or arr2.size == 0:
            return np.array([])

        result = np.array([])
        for x1 in arr1:
            for x2 in arr2:
                if all(x1 == x2):
                    result = self._union(result, np.array([x1]))
        return result

    def _relabel_nn(self, x):
        """
        Performs relabeling in the nearest neighborhood of x.

        :param x:
            An observation.
        :return:
        """
        neighborhood_candidates = self._knn(x, self._union(self.DS, self._union(self.AS,
                                                                                 self.RS)))  # do we actually need the union with DS and AS? We're taking intersect with RS in the next step anyway
        TS = self._intersect(self.RS,
                             neighborhood_candidates)  # TS - neighbors from majority class that can be relabeled
        while TS.shape[0] > 0 and \
                any(majority_class in self._min_cost_classes(x, self._union(self.DS, self._union(self.AS, self.RS)))
                    for majority_class in self.majority_classes):
            y = self._nearest(x, TS)
            TS = self._setdiff(TS, np.array([y]))
            self.RS = self._setdiff(self.RS, np.array([y]))
            y[-1] = x[-1]
            self.AS = self._union(self.AS, np.array([y]))

    @staticmethod
    def _nearest(x, TS):
        """
        Returns nearest neighbor of x in TS.

        :param x:
            Single observation.
        :param TS:
            Temporal set.
        :return:
            Nearest neighbor of x in TS.
        """
        clf = NearestNeighbors(n_neighbors=1).fit(TS[:, :-1])
        indices = clf.kneighbors([x[:-1]], return_distance=False)
        return TS[indices[0]][0]

    def _clean_nn(self, x):
        """
        Performs cleaning in the nearest neighborhood of x.

        :param x:
            Single observation.
        :return:
        """

        for majority_class in self.majority_classes:
            TS = self._knn(x, self._union(self.DS, self._union(self.AS, self.RS)), majority_class)
            while TS.shape[0] > 0 and \
                    majority_class in self._min_cost_classes(x, self._union(self.DS, self._union(self.AS, self.RS))):
                y = self._nearest(x, TS)
                TS = self._setdiff(TS, np.array([y]))
                self.DS = self._setdiff(self.DS, np.array([y]))
                self.RS = self._setdiff(self.RS, np.array([y]))

    def _knn(self, x, DS, c=None):
        """
        Returns k nearest neighbors of x in DS that belong to c class if specified.

        :param x:
            Single observation
        :param DS:
            DS
        :param c:
            Class of neighbors that should be returned.
        :return:
            These neighbors from k nearest that belong to class c if specified. Otherwise all of them.
        """

        self.neigh_clf.fit(DS[:, :-1])
        indices = self.neigh_clf.kneighbors([x[:-1]], return_distance=False)[0]
        if c is not None:
            result = []
            for idx in indices:
                if self._class_of(DS[idx]) == c:
                    result.append(DS[idx])
            return np.array(result)
        else:
            return DS[indices]

    def _amplify(self, x):
        """
        Artificially amplifies example x by adding a copy of it to the AS.

        :param x:
            Single observation.
        :return:
        """

        while self._class_of(x) not in self._min_cost_classes(x, self._union(self.DS, self._union(self.AS, self.RS))):
            y = x.copy()
            self.AS = self._union(self.AS, np.asarray([y]))

    @staticmethod
    
    def _class_of(example):
        return example[-1]


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
