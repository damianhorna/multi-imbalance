from collections import Counter
from typing import Tuple

import numpy as np
from imblearn.base import BaseSampler


class CCR(BaseSampler):
    """
    CCR is a combined cleaning and resampling energy-based algorithm.

    Each minority example has an associated energy budget that is used to expand a sphere around it. With each
    majority example within the sphere, the cost of further expansion increases. When energy is used up,
    majority examples are pushed out of the spheres and synthetic minority examples are generated inside the spheres.
    Synthetic examples are generated until the count of minority examples is approximately equal to the count of
    majority examples. Smaller spheres generate more synthetic examples than big ones to force the classification
    algorithm to focus on the most difficult examples.

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

        clean_majority, synthetic_minority = self._clean_and_generate(minority_examples, majority_examples)

        return np.vstack([minority_examples, clean_majority, synthetic_minority]), np.hstack([
            np.full((minority_examples.shape[0],), minority_class),
            y[y != minority_class],
            np.full((synthetic_minority.shape[0],), minority_class)
        ])

    def _clean_and_generate(self, minority_examples: np.ndarray, majority_examples: np.ndarray,
                            synthetic_examples_total: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param minority_examples:
            two-dimensional numpy array (number of samples x number of features) with float numbers of minority class
        :param majority_examples:
            two-dimensional numpy array (number of samples x number of features) with float numbers of majority class
        :param synthetic_examples_total:
            number of synthetic examples to be generated, if left as None it is calculated as difference of class counts
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

        generation_order = r.argsort()
        if synthetic_examples_total is None:
            synthetic_examples_total = majority_examples.shape[0] - minority_examples.shape[0]
        synthetic_examples_counts = (r ** -1 / (r ** -1).sum()) * synthetic_examples_total
        synthetic_leftovers = int((synthetic_examples_counts - synthetic_examples_counts.astype(int)).sum())
        synthetic_examples_counts = np.floor(synthetic_examples_counts).astype(int)

        for i in range(synthetic_leftovers):
            synthetic_examples_counts[generation_order[i % len(generation_order)]] += 1

        generated = []
        for i in generation_order:
            x = minority_examples[i]
            for j in range(synthetic_examples_counts[i]):
                random_translation = np.random.rand(majority_examples.shape[1]) * 2 - 1
                multiplier = random_translation / abs(random_translation).sum()
                new_point = x + multiplier * r[i] * np.random.rand(1)
                generated.append(new_point)

                if len(generated) == synthetic_examples_total:
                    break
            if len(generated) == synthetic_examples_total:
                break

        if len(generated) > 0:
            generated = np.array(generated)
        else:
            generated = np.empty((0, minority_examples.shape[1]))

        return clean_majority_examples, generated

    def distances(self, minority_example, majority_examples):
        return (abs(minority_example - majority_examples)).sum(1)


class MultiClassCCR(BaseSampler):
    """
    CCR for multi-class problems.

    The approach consists of the following steps:
    1. The classes are sorted in the descending order by the number of associated observations.
    2. For each of the minority classes, a collection of combined majority observations is constructed, consisting of
    a randomly sampled fraction of observations from each of the already considered class.
    3. Preprocessing with the CCR algorithm is performed, using the observations from the currently considered class
    as a minority, and the combined majority observations as the majority class. Both the generated synthetic
    minority observations and the applied translations are incorporated into the original data, and the synthetic
    observations can be used to construct the collection of combined majority observations for later classes.

    Koziarski, M., Wozniak, M., Krawczyk, B.: Combined Cleaning and Resampling Algorithm for Multi-Class Imbalanced
    Data with Label Noise. (2020)
    """

    def __init__(self, energy: float):
        """
        :param energy:
            initial energy budget for each minority example to use for sphere expansion
        """
        super().__init__()
        self._sampling_type = "over-sampling"
        self.CCR = CCR(energy=energy)

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
                clean_X_majority, synthetic_minority = self.CCR._clean_and_generate(X_minority, X_majority,
                                                                                    n_max - current_class_count)
                class_X[current_class] = np.vstack([class_X[current_class], synthetic_minority])
                clean_X_splits = [sample.shape[0] for _, sample in class_samples[:-1]]
                for j in range(1, len(clean_X_splits)):
                    clean_X_splits[j] += clean_X_splits[j - 1]
                split_clean_X = np.split(clean_X_majority, clean_X_splits)

                for j, (clazz, sample) in enumerate(class_samples):
                    class_X[clazz][sample] = split_clean_X[j]

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
