from typing import Tuple

import numpy as np
from imblearn.base import BaseSampler


class CCR(BaseSampler):

    def __init__(self, energy: float):
        super().__init__()
        self.energy = energy
        self._sampling_type = "over-sampling"

    def _fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        oversampled_X, oversampled_y = np.copy(X), np.copy(y)

        majority_examples = X[y == 0]
        minority_examples = X[y == 1]

        r = np.zeros(minority_examples.shape[0])
        e = np.full(minority_examples.shape[0], self.energy, dtype=float)
        t = np.zeros(majority_examples.shape)

        for i, x in enumerate(minority_examples):
            distances = self.distances(x, majority_examples)

            while e[i] > 0:
                examples_in_radius = distances <= r[i]
                nop = examples_in_radius.sum() + 1
                if nop == majority_examples.shape[0]+1:
                    nop -= 1
                    dr = e[i]/nop
                    r[i] += dr
                    break

                dr = e[i] / nop
                examples_in_radius2 = distances <= r[i] + dr
                nop2 = examples_in_radius2.sum() + 1
                if nop2 > nop:
                    examples_outside_radius = examples_in_radius2 ^ examples_in_radius
                    outside_index = np.flatnonzero(examples_outside_radius)
                    newdr = distances[outside_index].min() - r[i]
                    dr = newdr
                r[i] += dr
                e[i] -= dr * ((distances < r[i]).sum() + 1)
            examples_in_range_index = np.flatnonzero(distances <= r[i])
            for j in examples_in_range_index:
                translation = majority_examples[j] - x
                d = distances[j]
                t[j] += (r[i] - d)/d * translation
                test = t[j]

        oversampled_X[y == 0] += t



        G = majority_examples.shape[0] - minority_examples.shape[0]
        inverse_radius_sum = (r**-1).sum()

        generated = []
        for i, x in enumerate(minority_examples):
            g = int(np.round(r[i]**-1/inverse_radius_sum * G))
            for j in range(g):
                random_translation = np.random.rand(majority_examples.shape[1])*2-1
                multiplier = random_translation/abs(random_translation).sum()
                new_point = x+multiplier*r[i]*np.random.rand(1)
                generated.append(new_point)

        return np.concatenate([oversampled_X, generated]), np.concatenate([oversampled_y, [1 for x in generated]]), r

    def distances(self, minority_example, majority_examples):
        # distances = np.linalg.norm(minority_example - majority_examples, axis=-1, ord=1)
        return (abs(minority_example - majority_examples)).sum(1)

    def NoP(self, minority_example: np.ndarray, majority_examples: np.ndarray, radius: float):
        # distances = np.linalg.norm(minority_example - majority_examples, axis=-1, ord=1)
        distances = (abs(minority_example - majority_examples)).sum(1)
        return (distances < radius).sum() + 1
