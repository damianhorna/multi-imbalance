from typing import Tuple

import numpy as np
from imblearn.base import BaseSampler


class CCR(BaseSampler):
    """
    CCR is a combined cleaning and resampling energy-based algorithm.

    Each minority example has an associated energy budget that is used to expand a sphere around it.
    With each majority example within the sphere, the cost of further expansion increases.
    When energy is used up, majority examples are pushed out of the spheres and synthetic minority examples are generated inside the spheres.
    Synthetic examples are generated until the count of minority examples is approximately equal to the count of majority examples.
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
            one-dimensional numpy array with labels for rows in X
        :return:
            resampled X, resampled y
        """
        minority_class = min(list(Counter(y).items()), key=lambda x: x[1])[0]

        minority_examples = X[y == minority_class]
        majority_examples = X[y != minority_class]

        clean_majority, synthetic_minority = self.clean_and_generate(minority_examples, majority_examples)

        return np.vstack([minority_examples, clean_majority, synthetic_minority]), np.hstack([
            np.full((minority_examples.shape[0],), minority_class),
            y[y != minority_class],
            np.full((synthetic_minority.shape[0],), minority_class)
        ])

    def clean_and_generate(self, minority_examples: np.ndarray, majority_examples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param minority_examples:
            two-dimensional numpy array (number of samples x number of features) with float numbers of minority class
        :param majority_examples:
            two-dimensional numpy array (number of samples x number of features) with float numbers of majority class
        :return:
            clean majority X, synthetic minority X
        """
        clean_majority_examples = np.copy(majority_examples)

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
                d = distances[j]
                if d == 0:
                    continue
                translation = majority_examples[j] - x
                t[j] += (r[i] - d) / d * translation

        clean_majority_examples += t

        number_of_synthetic_examples = majority_examples.shape[0] - minority_examples.shape[0]
        inverse_radius_sum = (r ** -1).sum()

        generated = []
        for i, x in enumerate(minority_examples):
            synthetic_examples = int(np.round(r[i] ** -1 / inverse_radius_sum * number_of_synthetic_examples))
            for j in range(synthetic_examples):
                random_translation = np.random.rand(majority_examples.shape[1]) * 2 - 1
                multiplier = random_translation / abs(random_translation).sum()
                new_point = x + multiplier * r[i] * np.random.rand(1)
                generated.append(new_point)
        generated = np.vstack(generated)
        return clean_majority_examples, generated

    def distances(self, minority_example, majority_examples):
        return (abs(minority_example - majority_examples)).sum(1)
