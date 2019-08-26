from collections import Counter
from sklearn.neighbors import NearestNeighbors

import numpy as np


class SOUP(object):
    # TODO docs and tests
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        neigh = NearestNeighbors(n_neighbors=self.k)
        neigh.fit(X)

        quantities = Counter(y)
        mean_quantity = np.mean(list(quantities.values()))

        for class_name, class_quantity in quantities.items():
            class_indexes = [i for i, value in enumerate(y) if value == class_name]

            safe_levels = dict()
            for sample_id in class_indexes:
                neighbours_indices = neigh.kneighbors([list(X[sample_id])], return_distance=False)

                # TODO Calculate safe levels
                break
            if class_quantity < mean_quantity:
                # TODO oversampling
                pass
            else:
                # TODO undersampling
                pass
