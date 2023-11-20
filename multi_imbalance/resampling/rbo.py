import random
from collections import Counter
from typing import Tuple, Callable, Any

import numpy as np
from imblearn.base import BaseSampler


class RBO(BaseSampler):
    """
    Radial-based oversampling algorithm.

    Radial-based oversampling tries to overcome limitations of neighborhood-based oversampling algorithms by using
    class potential estimation with radial-basis function. Potential represents the cumulative proximity of an
    example to given collection of examples. Class potential is the potential computed on the collection of examples
    belonging to a specific class. Mutual class potential is a difference between the potential of two different
    classes at some point in the space of examples. New examples are generated in the areas of low mutual class
    potential by random walking from randomly selected minority examples.

    Reference:
    Krawczyk, B., Koziarski, M., Wozniak, M.: Radial-Based Oversampling for Multiclass Imbalanced Data Classification
    IEEE Transactions on Neural Networks and Learning Systems
    """

    def __init__(self, gamma: float, step: int, iterations: int, k: int,
                 distance_function: Callable[[np.ndarray, np.ndarray], np.ndarray] =
                 lambda x, y: np.linalg.norm(x - y, ord=1, axis=1)) -> None:
        """
        :param gamma:
            spread of radial basis function
        :param step:
            optimization step
        :param iterations:
            number of iterations per synthetic observation
        :param k:
            number of neighbors used for potential approximation
        :param distance_function:
            vectorized function that calculates distance between an example and array of examples, defaults to L1 norm
        """
        super().__init__()
        self.gamma = gamma
        self.step = step
        self.iterations = iterations
        self.k = k
        self.distance_function = distance_function
        self._sampling_type = "over-sampling"

    def _fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X:
            two-dimensional numpy array (number of samples x number of features) with float numbers
        :param y:
            one-dimensional numpy array with labels for rows in X
        :return:
            resampled X, resampled y
        """
        minority_class = min(list(Counter(y).items()), key=lambda x: x[1])[0]

        minority_examples = X[y == minority_class]
        majority_examples = X[y != minority_class]

        synthetic_examples = self._generate_examples(minority_examples, majority_examples, minority_class)

        return np.vstack([X, synthetic_examples]), np.hstack([y, np.full((len(synthetic_examples),), minority_class)])

    def _generate_examples(self, minority_examples: np.ndarray, majority_examples: np.ndarray, minority_class: Any) -> \
            np.ndarray:
        synthetic_examples = []

        X = np.vstack([minority_examples, majority_examples])
        y = np.hstack([np.full((len(minority_examples),), minority_class),
                       np.full((len(majority_examples),), 1 - minority_class)])
        self.feature_count = X.shape[1]

        k_sorted_nearest_neighbours = self._find_k_sorted_nearest_neighbours(minority_examples, X)

        while len(minority_examples) + len(synthetic_examples) < len(majority_examples):
            random_minority_index = random.randint(0, len(minority_examples) - 1)
            example_k_nearest_neighbours = k_sorted_nearest_neighbours[random_minority_index]
            X_nearest_majority, X_nearest_minority = self._get_nearest_majority_and_minority_neighbours(
                example_k_nearest_neighbours, minority_class, X, y)

            current_x = minority_examples[random_minority_index].copy()
            new_x = self._generate_minority_example(current_x, X_nearest_majority, X_nearest_minority)
            synthetic_examples.append(new_x)

        return np.array(synthetic_examples)

    def _get_nearest_majority_and_minority_neighbours(self, k_nearest_neighbours, minority_class, X, y):
        nearest_classes = y[k_nearest_neighbours]
        X_nearest_majority = X[k_nearest_neighbours[nearest_classes != minority_class]]
        X_nearest_minority = X[k_nearest_neighbours[nearest_classes == minority_class]]
        return X_nearest_majority, X_nearest_minority

    def _find_k_sorted_nearest_neighbours(self, minority_examples, X):
        return [np.argsort(self.distance_function(minority, X))[1:self.k + 1] for minority in minority_examples]

    def _generate_minority_example(self, current_x: np.ndarray, x_majority: np.ndarray, x_minority: np.ndarray):
        mutual_potential = self._mutual_class_potential(current_x, x_majority, x_minority)
        for i in range(self.iterations):
            new_x = self._perturb_x(current_x, self.feature_count)
            new_mutual_potential = self._mutual_class_potential(new_x, x_majority, x_minority)
            if new_mutual_potential < mutual_potential:
                current_x = new_x
                mutual_potential = new_mutual_potential
        return current_x

    def _perturb_x(self, current_x: np.ndarray, feature_count: int):
        direction = np.zeros(feature_count)
        direction[np.random.randint(feature_count)] = 1
        sign = -1 if random.randint(0, 1) else 1
        new_x = current_x + direction * sign * self.step
        return new_x

    def _mutual_class_potential(self, x: np.ndarray, x_majority: np.ndarray, x_minority: np.ndarray):
        return self._potential(x, x_majority) - self._potential(x, x_minority)

    def _potential(self, x: np.ndarray, collection: np.ndarray) -> float:
        distances = self.distance_function(x, collection)
        weights = np.exp(- (distances / self.gamma) ** 2)
        total_weight = weights.sum()
        return total_weight


class MultiClassRBO(BaseSampler):
    """
    RBO for multi-class problems.

    The approach consists of the following steps:
    1. The classes are sorted in the descending order by the number of associated observations.
    2. For each of the minority classes, a collection of combined majority observations is constructed, consisting of
    a randomly sampled fraction of observations from each of the already considered class.
    3. Preprocessing with the RBO algorithm is performed, using the observations from the currently considered class
    as a minority, and the combined majority observations as the majority class. Both the generated synthetic
    minority observations and the applied translations are incorporated into the original data, and the synthetic
    observations can be used to construct the collection of combined majority observations for later classes.

    Reference:
    Krawczyk, B., Koziarski, M., Wozniak, M.: Radial-Based Oversampling for Multiclass Imbalanced Data Classification
    IEEE Transactions on Neural Networks and Learning Systems
    """

    def __init__(self, gamma: float, step: float, iterations: int, k: int,
                 distance_function: Callable[[np.ndarray, np.ndarray], np.ndarray] =
                 lambda x, y: np.linalg.norm(x - y, axis=1)):
        """
        :param gamma:
            spread of radial basis function
        :param step:
            optimization step
        :param iterations:
            number of iterations per synthetic observation
        :param k:
            number of neighbors used for potential approximation
        :param distance_function:
            vectorized function that calculates distance between two vectors
        """
        super().__init__()
        self.gamma = gamma
        self.step = step
        self.iterations = iterations
        self.k = k
        self.distance_function = distance_function
        self._sampling_type = "over-sampling"
        self.RBO = RBO(gamma, step, iterations, k, distance_function)

    def _fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X:
            two-dimensional numpy array (number of samples x number of features) with float numbers
        :param y:
            one-dimensional numpy array with labels for rows in X, assumes minority class = 1 and majority class = 0
        :return:
            resampled X, resampled y
        """
        sorted_class_counts = sorted(list(Counter(y).items()), key=lambda x: x[1], reverse=True)
        n_max = sorted_class_counts[0][1]
        class_X = {clazz: X[y == clazz] for clazz, _ in sorted_class_counts}

        for i in range(1, len(sorted_class_counts)):
            current_class, current_class_count = sorted_class_counts[i]
            number_of_classes_with_higher_count = sum([1 for _, count in sorted_class_counts[:i] if count > current_class_count])
            if number_of_classes_with_higher_count > 0:
                X_minority = class_X[current_class]
                X_majority = []
                class_samples = []
                for clazz, _ in sorted_class_counts[:i]:
                    if clazz != current_class:
                        sampled_X = class_X[clazz]
                        sampled_size = sampled_X.shape[0]
                        sample_size = int(n_max / number_of_classes_with_higher_count)
                        sample_size = min(sample_size, sampled_size)
                        sample = np.random.choice(sampled_size, sample_size, replace=False)
                        class_samples.append((clazz, sample))
                        X_majority.append(sampled_X[sample])
                X_majority = np.concatenate(X_majority)

                generated = self.RBO._generate_examples(X_minority, X_majority, current_class)
                class_X[current_class] = np.vstack([class_X[current_class], generated])

        final_X = np.vstack([class_X[clazz] for clazz, _ in sorted_class_counts])
        final_y = np.hstack([np.full((class_X[clazz].shape[0],), clazz) for clazz, _ in sorted_class_counts])
        return final_X, final_y