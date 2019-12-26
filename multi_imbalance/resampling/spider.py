import numpy as np
from sklearn.neighbors import NearestNeighbors
from multi_imbalance.utils.array_util import (union, setdiff, contains)
from collections import Counter


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

    def __init__(self, k, majority_classes, intermediate_classes, minority_classes, cost=None):
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
        self.AS, self.RS = np.array([]), np.array([])

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
        self.stds, self.means = [1] * X.shape[1], [0] * X.shape[1]
        if self.cost is None:
            self.cost = self._estimate_cost_matrix(y)

        self._restart_perspective()
        self._calculate_weak_majority_examples()
        self._restore_perspective()
        self.DS = setdiff(self.DS, self.RS)
        int_classes, min_classes = self._sort_by_cardinality(y)

        for int_min_class in int_classes + min_classes:
            self.relabel(int_min_class)
            self.clean(int_min_class)
            self.amplify(int_min_class)

        self.DS = union(self.DS, self.AS)

        return self.DS[:, :-1], self.DS[:, -1]

    @staticmethod
    def _estimate_cost_matrix(y):
        """
        Method that estimates cost matrix automatically. For example given imbalance ratios of 1:2:6, the estimated
        matrix will be:
        [0 1 1
        2 0 1
        6 3 0]
        :param y: labels
        :return: cost matrix
        """
        class_cardinality = Counter(y)
        classes = list(class_cardinality.keys())
        cost = np.ones([len(classes), len(classes)])
        for i, (c1, card1) in enumerate(class_cardinality.items()):
            for j, (c2, card2) in enumerate(class_cardinality.items()):
                if j > i:
                    cost[i, j] = 1
                else:
                    cost[i, j] = card1 / card2
        np.fill_diagonal(cost, 0)
        return cost

    def _sort_by_cardinality(self, y):
        class_cardinality = Counter(y)
        # to ensure looping over classes with decreasing cardinality.
        int_classes = sorted(self.intermediate_classes, key=lambda clazz: -class_cardinality[clazz])
        min_classes = sorted(self.minority_classes, key=lambda clazz: -class_cardinality[clazz])
        return int_classes, min_classes

    def amplify(self, int_min_class):
        self._restart_perspective()
        int_min_ds = self.DS[self.DS[:, -1] == int_min_class]
        for x in int_min_ds:
            self._amplify_nn(x)
        self._restore_perspective()

    def clean(self, int_min_class):
        self._restart_perspective()
        int_min_ds = self.DS[self.DS[:, -1] == int_min_class]
        int_min_as = self._calc_int_min_as(int_min_class)
        for x in union(int_min_ds, int_min_as):
            self._clean_nn(x)
        self._restore_perspective()

    def relabel(self, int_min_class):
        self._restart_perspective()
        int_min_ds = self.DS[self.DS[:, -1] == int_min_class]
        for x in int_min_ds:
            self._relabel_nn(x)
        self._restore_perspective()

    def _restart_perspective(self):
        """
        Performs normalization over resampled dataset.
        :return:
        """
        for col in range(self._ds_as_rs_union().shape[1] - 1):
            self.stds[col] = self._ds_as_rs_union()[:, col].std()
            self.means[col] = self._ds_as_rs_union()[:, col].mean()

        for col in range(self._ds_as_rs_union().shape[1] - 1):
            if self.stds[col] == 0:
                self.stds[col] = 1e-6

        for dataset in [self.DS, self.RS, self.AS]:
            if dataset.shape[0] > 0:
                self._normalize(dataset)

    def _restore_perspective(self):
        """
        Denormalizes for further processing.
        :return:
        """
        for dataset in [self.DS, self.RS, self.AS]:
            if dataset.shape[0] > 0:
                self._denormalize(dataset)

    def _normalize(self, dataset):
        for col in range(dataset.shape[1] - 1):
            dataset[:, col] = (dataset[:, col] - self.means[col]) / (4 * self.stds[col])

    def _denormalize(self, dataset):
        for col in range(dataset.shape[1] - 1):
            dataset[:, col] = dataset[:, col] * self.stds[col] * 4 + self.means[col]

    def _calc_int_min_as(self, int_min_class):
        """
        Helper method to calculate examples form AS that belong to int_min_class parameter class.
        :param int_min_class:
            The class name (intermediate or minority).
        :return:
            Examples from AS that are belong to int_min_class.
        """

        if self.AS.size != 0:
            int_min_as = self.AS[self.AS[:, -1] == int_min_class]
        else:
            int_min_as = np.array([])
        return int_min_as

    def _calculate_weak_majority_examples(self):
        """
        Calculates weak majority examples and appends them to the RS set.
        :return:
        """

        for majority_class in self.majority_classes:
            majority_examples = self.DS[self.DS[:, -1] == majority_class]
            for x in majority_examples:
                if majority_class not in self._min_cost_classes(x, self.DS):
                    self.RS = union(self.RS, np.array([x]))

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

        C = self.minority_classes + self.intermediate_classes + self.majority_classes
        vals = []
        kneighbors = self._knn(x, DS)

        for cj in C:
            s = 0
            for ci in C:
                s += ((kneighbors[:, -1] == ci).astype(int).sum() / self.k) * self.cost[C.index(ci), C.index(cj)]
            vals.append(s)
        C = np.array(C)
        vals = np.array(vals)
        vals = np.round(vals, 6)
        return C[vals == vals[np.argmin(vals)]]

    def _relabel_nn(self, x):
        """
        Performs relabeling in the nearest neighborhood of x.

        :param x:
            An observation.
        :return:
        """
        nearest_neighbors = self._knn(x, self._ds_as_rs_union())
        for neighbor in nearest_neighbors:
            if contains(self.RS, neighbor) and self._class_of(neighbor) in self.majority_classes and self._class_of(
                    neighbor) in self._min_cost_classes(x, self._ds_as_rs_union()):
                self.RS = setdiff(self.RS, np.array([neighbor]))
                neighbor[-1] = x[-1]
                self.AS = union(self.AS, np.array([neighbor]))

    def _clean_nn(self, x):
        """
        Performs cleaning in the nearest neighborhood of x.

        :param x:
            Single observation.
        :return:
        """
        nearest_neighbors = self._knn(x, self._ds_as_rs_union())
        for neighbor in nearest_neighbors:
            if self._class_of(neighbor) in self.majority_classes and \
                    self._class_of(neighbor) in self._min_cost_classes(x, self._ds_as_rs_union()):
                self.DS = setdiff(self.DS, np.array([neighbor]))
                self.RS = setdiff(self.RS, np.array([neighbor]))

    def _knn(self, x, DS):
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

        DS = setdiff(DS, np.array([x]))
        if DS.shape[0] < self.k:
            self.neigh_clf = NearestNeighbors(n_neighbors=DS.shape[0])
        else:
            self.neigh_clf = NearestNeighbors(n_neighbors=self.k)

        self.neigh_clf.fit(DS[:, :-1])

        within_radius = self.neigh_clf.radius_neighbors([x[:-1]], radius=
        self.neigh_clf.kneighbors([x[:-1]], return_distance=True)[0][0][-1] + 0.0001 *
        self.neigh_clf.kneighbors([x[:-1]], return_distance=True)[0][0][-1], return_distance=True)

        unique_distances = np.unique(sorted(within_radius[0][0]))
        all_distances = within_radius[0][0]
        all_indices = within_radius[1][0]
        indices = []
        for dist in unique_distances:
            if len(indices) < self.k:
                indices += (all_indices[all_distances == dist]).tolist()

        return DS[indices]

    def _amplify_nn(self, x):
        """
        Artificially amplifies example x by adding a copy of it to the AS.

        :param x:
            Single observation.
        :return:
        """

        while self._class_of(x) not in self._min_cost_classes(x, self._ds_as_rs_union()):
            y = x.copy()
            self.AS = union(self.AS, np.asarray([y]))

    @staticmethod
    def _class_of(example):
        return example[-1]

    def _ds_as_rs_union(self):
        return union(self.DS, union(self.AS, self.RS))
