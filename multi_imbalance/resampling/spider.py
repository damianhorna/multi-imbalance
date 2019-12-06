from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from multi_imbalance.datasets import load_datasets
from collections import Counter
import pandas as pd
from IPython.core.display import display
from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from multi_imbalance.datasets import load_datasets
from multi_imbalance.resampling.SOUP import SOUP
from multi_imbalance.resampling.MDO import MDO
from multi_imbalance.resampling.GlobalCS import GlobalCS

from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE

import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

maj_int_min = {
    "1czysty-cut": {'maj': [0],'int': [2],'min': [1]},
    "2delikatne-cut": {'maj': [0],'int': [2],'min': [1]},
    "3mocniej-cut": {'maj': [0],'int': [2],'min': [1]},
    "4delikatne-bezover-cut": {'maj': [0],'int': [2],'min': [1]},
    "balance-scale": {'maj': [2, 1],'int': [],'min': [0]},
    "cleveland": {'maj': [0],'int': [1],'min': [2, 3, 4]},
    "cleveland_v2": {'maj': [0],'int': [],'min': [1,2,3]},
    "cmc": {'maj': [0, 2],'int': [],'min': [1]},
    "dermatology": {'maj': [0,],'int': [2, 1, 4, 3],'min': [5]},
    "glass": {'maj': [1, 0],'int': [3],'min': [5, 2, 4]},
    "hayes-roth": {'maj': [],'int': [],'min': [0,1, 2]},
    "new_ecoli": {'maj': [0],'int': [1,4],'min': [2, 3]},
    "new_led7digit": {'maj': [3, 5, 0, 2,4,1],'int': [],'min': []},
    "new_vehicle": {'maj': [1],'int': [],'min': [0, 2]},
    "new_winequality-red": {'maj': [0, 1],'int': [2],'min': [3]},
    "new_yeast": {'maj': [0, 1],'int': [8, 7],'min': [6,5,4,3,2]},
    "thyroid-newthyroid": {'maj': [0],'int': [],'min': [1,2]}
}


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
        figure_number = 0


        self.DS = np.append(X, y.reshape(y.shape[0], 1), axis=1)

        self.plot(f"GENERATED-{figure_number}_before_processing.png")
        figure_number += 1
        self.calculate_weak_majority_examples()
        self.DS = self._setdiff(self.DS, self.RS)

        self.plot(f"GENERATED-{figure_number}_after_processing_majority.png")
        figure_number += 1

        for int_min_class in self.intermediate_classes + self.minority_classes: ## TODO: kolejność klas zgodnie z malejącą licznością
            int_min_ds = self.DS[self.DS[:, -1] == int_min_class]
            for x in int_min_ds:
                self._relabel_nn(x)

            self.plot(f"GENERATED-{figure_number}_after_relabelling_to_{int_min_class}.png")
            figure_number += 1

            int_min_as = self.calc_int_min_as(int_min_class)
            for x in self._union(int_min_ds, int_min_as):
                self._clean_nn(x)

            self.plot(f"GENERATED-{figure_number}_after_cleaning_{int_min_class}.png")
            figure_number += 1

            for x in int_min_ds:
                self._amplify(x)

            self.plot(f"GENERATED-{figure_number}_after_amplifying_{int_min_class}.png")
            figure_number += 1

        self.DS = self._union(self.DS, self.AS)

        self.plot(f"GENERATED-{figure_number}_final.png", dataset=self.DS)

        return self.DS[:, :-1], self.DS[:, -1]

    def plot(self, path, dataset=None):
        if dataset is None:
            dataset = self.ds_as_rs_union()
        plt.figure(figsize=(12, 12))
        sns.scatterplot(x='x1', y='x2', hue='y', style='y',
                        data=pd.DataFrame(data=pd.DataFrame(data=dataset, columns=["x1", "x2", "y"]),
                                          columns=["x1", "x2", "y"]), alpha=0.7, legend=False)
        plt.savefig(path)

        spider_result = pd.read_csv(f"java_version/{path[:-4]}.csv")
        plt.figure(figsize=(12, 12))
        sns.scatterplot(x='X1', y='X2', hue='CLASS', style='CLASS', data=spider_result, alpha=0.7, legend=False)
        plt.savefig(f"{path[:-4]}_spider.png")

    def calc_int_min_as(self, int_min_class):
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

        C = self.minority_classes + self.intermediate_classes + self.majority_classes
        vals = []
        kneighbors = self._knn(x, DS)

        for cj in C:
            s = 0
            for ci in C:
                s += ((kneighbors[:,-1] == ci).astype(int).sum() / self.k) * self.cost[C.index(ci), C.index(cj)]
            vals.append(s)
        C = np.array(C)
        vals = np.array(vals)
        vals = np.round(vals, 6)
        return C[vals == vals[np.argmin(vals)]]

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

        arr2tolist = arr2.tolist()
        arr1tolist = arr1.tolist()
        for element in arr2tolist:
            if element in arr1tolist:
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
        nearest_neighbors = self._knn(x, self.ds_as_rs_union())
        TS = self._intersect(self.RS, nearest_neighbors)
        while TS.shape[0] > 0 and \
                any(majority_class in self._min_cost_classes(x, self.ds_as_rs_union())
                    for majority_class in self.majority_classes):
            y = self._nearest(x, TS)
            TS = self._setdiff(TS, np.array([y]))
            self.RS = self._setdiff(self.RS, np.array([y]))
            y[-1] = x[-1]
            self.AS = self._union(self.AS, np.array([y]))

    def _nearest(self, x, TS):
        """
        Returns nearest neighbor of x in TS.

        :param x:
            Single observation.
        :param TS:
            Temporal set.
        :return:
            Nearest neighbor of x in TS.
        """
        TS = self._setdiff(TS, np.array([x]))
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
        nearest_neighbors = self._knn(x, self.ds_as_rs_union())
        for neighbor in nearest_neighbors:
            if self._class_of(neighbor) in self.majority_classes and \
                    self._class_of(neighbor) in self._min_cost_classes(x, self.ds_as_rs_union()):
                self.DS = self._setdiff(self.DS, np.array([neighbor]))
                self.RS = self._setdiff(self.RS, np.array([neighbor]))

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

        DS = self._setdiff(DS, np.array([x]))
        if DS.shape[0] < self.k:
            self.neigh_clf = NearestNeighbors(n_neighbors=DS.shape[0])
        else:
            self.neigh_clf = NearestNeighbors(n_neighbors=self.k)

        self.neigh_clf.fit(DS[:, :-1])
        within_radius = self.neigh_clf.radius_neighbors([x[:-1]], radius=self.neigh_clf.kneighbors([x[:-1]], return_distance=True)[0][0][-1] + 0.0001 * self.neigh_clf.kneighbors([x[:-1]], return_distance=True)[0][0][-1],return_distance=True)
        unique_distances = np.unique(sorted(within_radius[0][0]))
        all_distances = within_radius[0][0]
        all_indices = within_radius[1][0]
        indices = []
        for dist in unique_distances:
            if len(indices) < self.k:
                indices += (all_indices[all_distances == dist]).tolist()

        return DS[indices]

    def _amplify(self, x):
        """
        Artificially amplifies example x by adding a copy of it to the AS.

        :param x:
            Single observation.
        :return:
        """

        while self._class_of(x) not in self._min_cost_classes(x, self.ds_as_rs_union()):
            y = x.copy()
            self.AS = self._union(self.AS, np.asarray([y]))

    @staticmethod
    def _class_of(example):
        return example[-1]

    def ds_as_rs_union(self):
        return self._union(self.DS, self._union(self.AS, self.RS))


def read_train_and_test_data(overlap, imbalance_ratio, i):
    with open(f"../../../3class-ho/3class-{imbalance_ratio}-overlap-{overlap}-learn-{i}.arff") as f:
        content = f.readlines()
    content = [x.strip().split(",") for x in content][5:]
    data = np.array(content)
    X_train, y_train = data[:, :-1].astype(float), data[:, -1].astype(object)

    with open(f"../../../3class-ho/3class-{imbalance_ratio}-overlap-{overlap}-test-{i}.arff") as f:
        content = f.readlines()
    content = [x.strip().split(",") for x in content][5:]
    data = np.array(content)
    X_test, y_test = data[:, :-1].astype(float), data[:, -1].astype(object)

    return X_train, y_train, X_test, y_test


def train_and_test():
    neigh = KNeighborsClassifier(n_neighbors=1)
    # for i in range(0, 2):
    #     X_train[:, i] = (X_train[:, i] - np.mean(X_train[:, i])) / np.std(X_train[:, i])
    #     X_test[:, i] = (X_test[:, i] - np.mean(X_test[:, i])) / np.std(X_test[:, i])
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    labels = ['MIN', 'INT', 'MAJ']
    # for i, label in enumerate(labels):
    #     print(
    #         f"{label} TPR: {confusion_matrix(y_test, y_pred, labels=labels)[i, i] / confusion_matrix(y_test, y_pred, labels=labels)[:, i].sum()}")
    return [confusion_matrix(y_test, y_pred, labels=labels)[i, i] / confusion_matrix(y_test, y_pred, labels=labels)[i,
                                                                    :].sum() for i, label in enumerate(labels)]


if __name__ == "__main__":
    tprs = []
    for imbalance_ratio in [ "30-40-15-15"]:  # "70-30-0-0", "40-50-10-0",
        print(f"Imbalance ratio: {imbalance_ratio}")
        for overlap in [0]:  # , 1, 2
            print(f"Overlap: {overlap}")
            min_tpr = []
            int_tpr = []
            maj_tpr = []
            for i in range(1, 2): # 11
                X_train, y_train, X_test, y_test = read_train_and_test_data(overlap, imbalance_ratio, i)
                cost = np.ones((3, 3))
                for i in range(3):
                    cost[i][i] = 0

                cost = np.reshape(np.array([0, 2, 3, 3, 0, 2, 7, 5, 0]), (3, 3))
                #cost = np.reshape(np.array([0, 1, 1, 1, 0, 1, 1, 1, 0]), (3, 3))

                clf = SPIDER3(k=5, cost=cost, majority_classes=['MAJ'],
                              intermediate_classes=['INT'], minority_classes=['MIN'])
                for k in range(0, 2):
                    X_train[:, k] = (X_train[:, k] - np.mean(X_train[:, k])) / np.std(X_train[:, k])
                    X_test[:, k] = (X_test[:, k] - np.mean(X_test[:, k])) / np.std(X_test[:, k])
                X_train, y_train = clf.fit_transform(X_train.astype(np.float64), y_train)
                min_t, int_t, maj_t = train_and_test()
                min_tpr.append(min_t)
                int_tpr.append(int_t)
                maj_tpr.append(maj_t)
            tprs.append([np.array(min_tpr).mean(), np.array(int_tpr).mean(), np.array(maj_tpr).mean()])
            print(f"MIN TPR:{np.array(min_tpr).mean()}")
            print(f"INT TPR:{np.array(int_tpr).mean()}")
            print(f"MAJ TPR:{np.array(maj_tpr).mean()}")
    np.savetxt("costs.csv", np.array(tprs), delimiter=",")





if __name__ == "__main__2":
    datasets = load_datasets()
    results_g_mean = dict()
    results_acc = dict()

    for dataset_name, dataset_values in datasets.items():
        if dataset_name != 'dermatology': #or dataset_name != 'new_ecoli':
            continue
        print(dataset_name)

        X, y = dataset_values.data, dataset_values.target

        if len(X) > 1000:
            continue

        results_g_mean[dataset_name] = dict()
        results_acc[dataset_name] = dict()

        for resample in ['base', 'global', 'soup', 'mdo', 'spider']:

            skf = StratifiedKFold(n_splits=5, random_state=0)
            acc, g_mean = list(), list()
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                error_flag = False
                clf_tree = DecisionTreeClassifier(random_state=0)

                if resample == 'base':
                    X_train_resampled, y_train_resampled = X_train, y_train
                elif resample == 'soup':
                    soup = SOUP()
                    X_train_resampled, y_train_resampled = soup.fit_transform(np.copy(X_train), np.copy(y_train))
                elif resample == 'global':
                    global_cs = GlobalCS()
                    X_train_resampled, y_train_resampled = global_cs.fit_transform(np.copy(X_train), np.copy(y_train))
                elif resample == 'smote':
                    try:
                        smote = SMOTE()
                        X_train_resampled, y_train_resampled = smote.fit_sample(np.copy(X_train), np.copy(y_train))
                    except Exception as e:
                        error_flag = True
                        print(resample, dataset_name, e)
                        X_train_resampled, y_train_resampled = X_train, y_train
                elif resample == 'mdo':
                    mdo = MDO(k=9, k1_frac=0, seed=0)
                    X_train_resampled, y_train_resampled = mdo.fit_transform(np.copy(X_train), np.copy(y_train))
                elif resample == 'spider':
                    cost = calc_cost_matrix(dataset_name)
                    clf = SPIDER3(k=5, cost=cost, majority_classes=maj_int_min[dataset_name]['maj'],
                                  intermediate_classes=maj_int_min[dataset_name]['int'],
                                  minority_classes=maj_int_min[dataset_name]['min'])
                    X_train_resampled, y_train_resampled = clf.fit_transform(X_train.astype(np.float64), y_train)

                clf_tree.fit(X_train_resampled, y_train_resampled)
                y_pred = clf_tree.predict(X_test)
                g_mean.append(geometric_mean_score(y_test, y_pred, correction=0.001))
                acc.append(accuracy_score(y_test, y_pred))

            result_g_mean = None if error_flag else round(np.mean(g_mean), 3)
            result_acc = None if error_flag else round(np.mean(acc), 3)

            results_g_mean[dataset_name][resample] = result_g_mean
            results_acc[dataset_name][resample] = result_acc

    display("G-MEAN")
    df = pd.DataFrame(results_g_mean).T
    display(df)

    # display("ACC")
    # df2 = pd.DataFrame(results_acc).T
    # display(df2)

    display("MEAN G-MEAN")
    df.fillna(df.median(), inplace=True)
    display(df.mean())

    # display("MEAN ACC")
    # display(df2.mean())

