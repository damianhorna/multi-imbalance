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
                 lambda x, y: np.linalg.norm(x - y, axis=1)) -> None:
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

        S = self._generate_examples(minority_examples, majority_examples, minority_class)

        return np.vstack([X, S]), np.hstack([y, np.full((len(S),), minority_class)])

    def _generate_examples(self, minority_examples: np.ndarray, majority_examples: np.ndarray, minority_class: Any) -> \
            np.ndarray:
        S = []

        X = np.vstack([minority_examples, majority_examples])
        y = np.hstack([np.full((len(minority_examples),), minority_class),
                       np.full((len(majority_examples),), 1 - minority_class)])
        feature_count = X.shape[1]

        minority_nearest = [np.argsort((abs(minority - X)).sum(1))[1:self.k + 1] for minority in minority_examples]

        while len(minority_examples) + len(S) < len(majority_examples):
            random_minority_index = random.randint(0, len(minority_examples) - 1)
            current_minority_example = minority_examples[random_minority_index].copy()
            nearest_index = minority_nearest[random_minority_index]
            nearest_classes = y[nearest_index]
            x_majority = X[nearest_index[nearest_classes != minority_class]]
            x_minority = X[nearest_index[nearest_classes == minority_class]]

            mutual_potential = self._potential(current_minority_example, x_majority) - self._potential(
                current_minority_example, x_minority)

            for i in range(self.iterations):
                direction = np.zeros(feature_count)
                direction[np.random.randint(feature_count)] = 1
                sign = -1 if random.randint(0, 1) else 1
                new_x = current_minority_example + direction * sign * self.step
                new_potential = self._potential(new_x, x_majority) - self._potential(new_x, x_minority)
                if new_potential < mutual_potential:
                    current_minority_example = new_x
                    mutual_potential = self._potential(current_minority_example, x_majority) - self._potential(
                        current_minority_example, x_minority)

            S.append(current_minority_example)
        return np.array(S)

    def _potential(self, example: np.ndarray, collection: np.ndarray) -> float:
        distances = self.distance_function(example, collection)
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

    def __init__(self, gamma: float, step: int, iterations: int, k: int,
                 distance_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda x, y: np.linalg.norm(x - y,
                                                                                                                 axis=1)):
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
            number_of_classes_with_higher_count = self._number_of_classes_with_higher_count(sorted_class_counts, i)
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

    def _number_of_classes_with_higher_count(self, sorted_class_counts, i):
        number_of_classes_with_higher_count = 0
        _, current_class_count = sorted_class_counts[i]
        for _, class_count in sorted_class_counts[:i]:
            if class_count > current_class_count:
                number_of_classes_with_higher_count += 1
        return number_of_classes_with_higher_count
