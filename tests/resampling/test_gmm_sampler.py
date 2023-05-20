import logging
import numpy as np
import pytest
from collections import Counter, OrderedDict

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
    def _get_parametrized_gmm_sampler(*args, **kwargs) -> GMMSampler:
        gmm_sampler = GMMSampler(*args, **kwargs)
        return gmm_sampler

    return _get_parametrized_gmm_sampler


def get_goal_quantity(y):
    quantities = Counter(y)
    return np.mean((quantities[minority_class], quantities[majority_class]), dtype=int)


@pytest.mark.parametrize("strategy, filter_new", [("median", 1.0), ("average", 0.0)])
@pytest.mark.parametrize("X, y", complete_test_data)
def test_output_length_with_undersample(X, y, strategy, filter_new, gmm_sampler_mock):
    gmm_sampler = gmm_sampler_mock(undersample=True, strategy=strategy, filter_new=filter_new)
    resampled_X, resampled_y = gmm_sampler.fit_resample(X, y)

    y_resampled_count = Counter(resampled_y)
    for quantity in y_resampled_count.values():
        assert quantity == get_goal_quantity(y)

    assert len(resampled_X) == get_goal_quantity(y) * num_classes
    assert len(resampled_y) == get_goal_quantity(y) * num_classes


@pytest.mark.parametrize("X, y", complete_test_data)
def test_output_length_without_undersample(X, y, gmm_sampler_mock):
    gmm_sampler = gmm_sampler_mock(undersample=False)
    _, resampled_y = gmm_sampler.fit_resample(X, y)

    y_count = Counter(y)
    y_resampled_count = Counter(resampled_y)

    assert y_resampled_count[minority_class] == get_goal_quantity(y)
    assert y_resampled_count[majority_class] == y_count[majority_class]


def test_perform_step_condition(gmm_sampler_mock):
    gmm_sampler = gmm_sampler_mock()
    assert gmm_sampler._perform_step(n_components=2, likelihood=1.0, num_samples=3)
    assert not gmm_sampler._perform_step(n_components=2, likelihood=-1.0, num_samples=3)
    assert not gmm_sampler._perform_step(n_components=2, likelihood=1.0, num_samples=1)
    gmm_sampler.max_components = 1
    assert not gmm_sampler._perform_step(n_components=4, likelihood=1.0, num_samples=3)


def test_minority_classes(gmm_sampler_mock):
    minority_classes = [0, 1]
    gmm_sampler = gmm_sampler_mock(minority_classes=minority_classes)

    assert gmm_sampler.minority_classes == minority_classes


@pytest.mark.parametrize(
    "maj_int_min, expected_size",
    [
        ({"maj": [], "int": [], "min": [1]}, 6),
        ({"maj": [1], "int": [], "min": []}, 6),
        ({"maj": [], "int": [1], "min": []}, 6),
    ],
)
def test_set_size_to_align(gmm_sampler_mock, expected_size, maj_int_min):
    gmm_sampler = gmm_sampler_mock()
    gmm_sampler.class_sizes = Counter(y_imb_hard)
    gmm_sampler.maj_int_min = OrderedDict(maj_int_min)

    gmm_sampler._set_size_to_align()
    assert gmm_sampler.size_to_align == expected_size


def test_compute_mdist(gmm_sampler_mock, caplog):
    caplog.set_level(logging.ERROR)
    gmm_sampler = gmm_sampler_mock()
    mean = [0, 0]
    covariance = np.eye(2)
    x = np.random.multivariate_normal(mean, covariance, size=2)

    gmm_sampler._compute_mdist(x, mean, np.ones((2, 2)))

    no_exception_check = 0
    for record in caplog.records:
        if record.levelname == "ERROR":
            msg = record.message
            no_exception_check += (
                msg == "Can't compute 'cdist' function. Distance threshold is set to 2.0"
                or msg == "For more information, examine exception: Singular matrix"
            )

    assert no_exception_check == 2


@pytest.mark.parametrize(
    "strategy, class_count, expected_middle_size",
    [("median", [10, 6, 4], 6), ("median", [4, 12, 4], 4), ("average", [4, 4, 12], 6), ("average", [8, 4, 8], 6)],
)
def test_get_middle_size_based_on_strategy(strategy, class_count, expected_middle_size, gmm_sampler_mock):
    gmm_sampler = gmm_sampler_mock(strategy=strategy)

    gmm_sampler._fit(X, np.array([*[0] * class_count[0], *[1] * class_count[1], *[2] * class_count[2]]))
    middle_size = gmm_sampler._get_middle_size_based_on_strategy()
    assert middle_size == expected_middle_size


def test_get_middle_size_based_on_strategy_exception(gmm_sampler_mock):
    gmm_sampler = gmm_sampler_mock()
    gmm_sampler.strategy = "min"

    with pytest.raises(ValueError) as ex:
        gmm_sampler._get_middle_size_based_on_strategy()

    assert str(ex.value) == 'Unrecognized min. Only "median" and "average" are allowed.'


def test_set_size_to_align_exception(gmm_sampler_mock):
    maj_int_min = {"maj": [], "int": [], "min": []}
    gmm_sampler = gmm_sampler_mock()
    gmm_sampler.class_sizes = Counter(y_imb_hard)
    gmm_sampler.maj_int_min = OrderedDict(maj_int_min)

    with pytest.raises(ValueError) as ex:
        gmm_sampler._set_size_to_align()

    assert str(ex.value) == "Bad input - can not obtain desire size."
