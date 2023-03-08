import numpy as np
import pytest
from collections import Counter

from multi_imbalance.resampling.gmm_sampler import GMMSampler

X = np.array(
    [
        [0.05837771, 0.57543339],
        [0.06153624, 0.99871925],
        [0.14308529, 0.00681144],
        [0.23401697, 0.21188708],
        [0.2418553, 0.02137086],
        [0.32480534, 0.81547632],
        [0.42478482, 0.31995162],
        [0.50726834, 0.72621157],
        [0.54580968, 0.58025914],
        [0.55748531, 0.71866238],
        [0.69208769, 0.63759459],
        [0.70797377, 0.16348051],
        [0.76410615, 0.70451542],
        [0.81680686, 0.50793884],
        [0.8490789, 0.53826627],
        [0.8847505, 0.96856011],
        [0.9287003, 0.97580299],
        [0.9584236, 0.10536541],
        [0.96983103, 0.87666093],
        [0.97352367, 0.78807909],
    ]
)

majority_class = 0
minority_class = 1
num_classes = 2

y_balanced = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
y_imb_easy = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])
y_imb_hard = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0])
complete_test_data = [
    (X, y_balanced),
    (X, y_imb_easy),
    (X, y_imb_hard),
]


@pytest.fixture()
def gmm_sampler_mock():
    def _get_parametrized_gmm_sampler(X, y, undersample):
        gmm_sampler = GMMSampler(undersample=undersample)
        return gmm_sampler

    return _get_parametrized_gmm_sampler


def get_goal_quantity(y):
    quantities = Counter(y)
    return np.mean((quantities[minority_class], quantities[majority_class]), dtype=int)


@pytest.mark.parametrize("X, y", complete_test_data)
def test_output_length_with_undersample(X, y, gmm_sampler_mock):
    gmm_sampler = gmm_sampler_mock(X, y, True)
    resampled_X, resampled_y = gmm_sampler.fit_resample(X, y)

    y_resampled_count = Counter(resampled_y)
    for _, quantity in y_resampled_count.items():
        assert quantity == get_goal_quantity(y)

    assert len(resampled_X) == get_goal_quantity(y) * num_classes
    assert len(resampled_y) == get_goal_quantity(y) * num_classes


@pytest.mark.parametrize("X, y", complete_test_data)
def test_output_length_without_undersample(X, y, gmm_sampler_mock):
    gmm_sampler = gmm_sampler_mock(X, y, False)
    resampled_X, resampled_y = gmm_sampler.fit_resample(X, y)

    y_count = Counter(y)
    y_resampled_count = Counter(resampled_y)

    assert y_resampled_count[minority_class] == get_goal_quantity(y)
    assert y_resampled_count[majority_class] == y_count[majority_class]


def test_perform_step_condition(gmm_sampler_mock):
    gmm_sampler = GMMSampler()
    assert gmm_sampler._perform_step(n_components=2, likelihood=1.0, num_samples=3)
    assert not gmm_sampler._perform_step(n_components=2, likelihood=-1.0, num_samples=3)
    assert not gmm_sampler._perform_step(n_components=2, likelihood=1.0, num_samples=1)
    gmm_sampler.max_components = 1
    assert not gmm_sampler._perform_step(n_components=4, likelihood=1.0, num_samples=3)
