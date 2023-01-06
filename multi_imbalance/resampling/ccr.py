from typing import Tuple

import numpy as np
from imblearn.base import BaseSampler


class CCR(BaseSampler):
    """
    CCR is a combined cleaning and resampling energy-based algorithm.

    Each minority example has an associated energy budget that is used to expand a sphere around it.
    With each majority example within the sphere, the cost of further expansion increases.
    When energy is used up, majority examples are pushed out of the spheres and synthetic minority examples are generated inside the spheres.
    Synthetic examples are generated until the count of minority examples is equal to the count of majority examples.
    Smaller spheres generate more synthetic examples than big ones to force the classification algorithm to focus on the most difficult examples.

    Reference:
    Koziarski, M., Wozniak, M.: CCR: A combined cleaning and resampling algorithm for imbalanced data classification.
    International Journal of Applied Mathematics and Computer Science 2017
    """

    def __init__(self, energy: float):
        """
        :param energy:
            initial energy budget for each minority example to use for sphere expansion
        """
        super().__init__()
        self.energy = energy
        self._sampling_type = "over-sampling"

    def _fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X:
            two-dimensional numpy array (number of samples x number of features) with float numbers
        :param y:
            one-dimensional numpy array with labels for rows in X, assumes minority class = 1 and majority class = 0
        :return:
            resampled X, resampled y
        """
        oversampled_X, oversampled_y = np.copy(X), np.copy(y)

        majority_examples = X[y == 0]
        minority_examples = X[y == 1]

        majority_count = len(majority_examples)

        r = np.zeros(minority_examples.shape[0])
        t = np.zeros(majority_examples.shape)

        for i, x in enumerate(minority_examples):
            distances = self.distances(x, majority_examples)
            sorted_distances_index = np.argsort(distances)
            energy = self.energy
            current_example = 0
            number_of_points_in_radius = 1

            while current_example != majority_count and energy > 0:
                example_distance_index = sorted_distances_index[current_example]
                distance = distances[example_distance_index]
                if distance <= r[i]:
                    number_of_points_in_radius += 1
                dr = energy / number_of_points_in_radius

                shortest_distance = distances[sorted_distances_index[current_example]]
                if r[i] + dr >= shortest_distance:
                    number_of_points_in_radius += 1
                    dr = shortest_distance - r[i]

                r[i] += dr
                energy -= dr * (current_example + 1.0)
                current_example += 1

            if energy > 0:
                r[i] += energy / (number_of_points_in_radius - 1)

            examples_in_range_index = np.flatnonzero(distances <= r[i])
            for j in examples_in_range_index:
                translation = majority_examples[j] - x
                d = distances[j]
                t[j] += (r[i] - d) / d * translation

        oversampled_X[y == 0] += t

        G = majority_examples.shape[0] - minority_examples.shape[0]
        inverse_radius_sum = (r ** -1).sum()

        generated = []
        for i, x in enumerate(minority_examples):
            g = int(np.round(r[i] ** -1 / inverse_radius_sum * G))
            for j in range(g):
                random_translation = np.random.rand(majority_examples.shape[1]) * 2 - 1
                multiplier = random_translation / abs(random_translation).sum()
                new_point = x + multiplier * r[i] * np.random.rand(1)
                generated.append(new_point)

        return np.concatenate([oversampled_X, generated]), np.concatenate([oversampled_y, [1 for x in generated]])

    def distances(self, minority_example, majority_examples):
        return (abs(minority_example - majority_examples)).sum(1)
