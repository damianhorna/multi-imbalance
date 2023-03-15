from collections import OrderedDict, Counter
from copy import deepcopy
from typing import Optional, List, Dict, Tuple, Any, TypeVar

import logging
import numpy as np
from imblearn.over_sampling.base import BaseSampler
from imblearn.utils import Substitution
from imblearn.utils._docstring import _random_state_docstring
from pydantic import validate_arguments
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

GMMS = TypeVar("GMMS", bound="GMMSampler")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
)


@Substitution(
    random_state=_random_state_docstring,
)
class GMMSampler(BaseSampler):
    """
    GGMSampling algorithm that uses creating new examples by sampling from a multivariate normal distribution
    (whose parameters are estimated from the input data) and removing troublesome examples from the majority class.

    Parameters
    ----------
    likelihood_threshold : float, default=0.0
        Minimum likelihood change threshold. A value below this threshold will be equivalent to no change.

    k_neighbors : int, default=7
        The number of analyzed nearest neighbors during the analysis.
        Used during both undersampling and oversampling.

    undersample : bool, default=True
        A binary value indicating whether to perform an undersampling operation on majority classes.

    min_components : int, default=1
        Minimum number of components of GaussianMixture.

    max_components : Optional[int], default=None
        Maximum number of components of GaussianMixture. Without upper bound if not specified.

    minority_classes : Optional[List[int]], default=None
        List containing minority classes given by hand - no auto detection of minority classes will be done.

    valid_size : float, default=0.25
        Size of validation set to perform test for components choosing.

    filter_new : float, default=-1
        Parameter controlling the behavior after the oversampling operation.
        Checks if and how to filter newly created examples:
        -1 -> do not filter out
        0 -> filter out by max/mean value of created examples
        >0 -> specify your own value e.g. 2.0

    add_after_filtration : bool, default=True
        Value specifying whether to regenerate the examples after filtering.

    iterations_after_filtration : int, default=50
        This value will potentially avoid an endless loop of deleting and re-generating examples.
        The upper limit for the number of repetitions.

    covariance_type : "full", "tied", "diag", "spherical", default="full"
        String describing the type of covariance parameters to use in GaussianMixture. Must be one of:
        - "full"
            each component has its own general covariance matrix
        - "tied"
            all components share the same general covariance matrix
        - "diag"
            each component has its own diagonal covariance matrix
        - "spherical"
            each component has its own single variance

    strategy : str "average" or "median", default="average"
        The strategy of selecting the number of examples considers the target number in each class.

    {random_state}

    n_init : int, default=10
        The number of initializations to perform in GaussianMixture. The best results are kept.

    tol : float, default=1e-3
        The convergence threshold in GaussianMixture. EM iterations will stop when the lower bound
        average gain is below this threshold.

    max_iter : int, default=100
        The number of EM iterations to perform in GaussianMixture.

    Attributes
    ----------
    likelihoods : dict
        Likelihood of each minority class obtained after fitting the final Gaussian model.

    gaussian_mixtures : dict
        Dictionary containing all Gaussian models for each minority class separately.

    class_sizes : Optional[Counter]
        A dictionary containing the counts of each class.

    neighborhood : Optional[dict]
        Dictionary with information on the nearest points for each example separately.

    maj_int_min : OrderedDict
        A dictionary containing information on which class can be considered majority,
        which minority and which remaining class - a heuristic matching.

    size_to_align : Optional[np.ndarray]
        ndarray containing information about the quantity considered the gold standard -
        it is this size that we will want to generate and remove examples.

    cdist_min_count : int
        The minimum number of examples found in the data sample on which
        distances between points are calculated (by the cdist method).

    Examples
    --------
    >>> import numpy as np
    >>> from algorithms.gmm_sampler import GMMSampler
    >>> from sklearn.datasets import make_blobs
    >>> from collections import Counter
    >>> blobs = [800, 100]
    >>> X, y  = make_blobs(blobs, centers=[(-4, 0), (0,0)])
    # Make this a binary classification problem
    >>> y = np.array(y == 1, dtype=int)
    >>> gmm_sampler = GMMSampler()
    >>> X_res, y_res = gmm_sampler.fit_resample(X, y)
    >>> print('Class distribution before GMMsampling: %s' % Counter(y))
    >>> print(f'Class distribution after GMMsampling: %s' % Counter(y_res))
    Class distribution before GMMsampling: Counter({{0: 800, 1: 100}})
    Class distribution after GMMsampling: Counter({{0: 450, 1: 450}})
    # Without undersampling
    >>> gmm_sampler = GMMSampler(undersample=False)
    >>> X_res, y_res = gmm_sampler.fit_resample(X, y)
    >>> print('Class distribution before GMMsampling: %s' % Counter(y))
    >>> print(f'Class distribution after GMMsampling: %s' % Counter(y_res))
    Class distribution before GMMsampling: Counter({{0: 800, 1: 100}})
    Class distribution after GMMsampling: Counter({{0: 800, 1: 450}})
    """

    _sampling_type = "over-sampling"

    @validate_arguments
    def __init__(
        self,
        likelihood_threshold: float = 0.0,
        k_neighbors: int = 7,
        undersample: bool = True,
        min_components: int = 1,
        max_components: Optional[int] = None,
        minority_classes: Optional[List[int]] = None,
        valid_size: float = 0.25,
        filter_new: float = -1.0,
        add_after_filtration: bool = True,
        iterations_after_filtration: int = 50,
        strategy: str = "average",
        covariance_type: str = "full",
        n_init: int = 10,
        tol: float = 1e-3,
        max_iter: int = 100,
        random_state: Optional[int] = None,
    ):
        super().__init__(sampling_strategy="auto")
        self.likelihood_threshold = likelihood_threshold
        self.k_neighbors = k_neighbors
        self.undersample = undersample
        self.min_components = min_components
        self.max_components = max_components
        self._minority_classes = minority_classes
        self.valid_size = valid_size
        self.filter_new = filter_new
        self.add_after_filtration = add_after_filtration
        self.iterations_after_filtration = iterations_after_filtration
        self.n_init = n_init
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(self.random_state)

        assert strategy in ["average", "median"], f"strategy '{strategy}' is invalid."
        self.strategy = strategy

        assert covariance_type in [
            "full",
            "tied",
            "diag",
            "spherical",
        ], f"covariance_type '{covariance_type}' is invalid."
        self.covariance_type = covariance_type

        self.likelihoods: Dict[int, float] = dict()
        self.gaussian_mixtures: Dict[int, GaussianMixture] = dict()
        self.class_sizes: Optional[Counter[int]] = None
        self.neighborhood: Optional[Dict[int, float]] = None
        self.maj_int_min = OrderedDict({"maj": list(), "int": list(), "min": list()})
        self.size_to_align: Optional[int] = None
        self.__x_subset: Optional[np.ndarray] = None
        self.cdist_min_count = 10
        self.__logger = logging.getLogger("GMMSampler")

    @property
    def minority_classes(self) -> List[int]:
        if (self.class_sizes is None) or (self._minority_classes is not None):
            return self._minority_classes
        return self.maj_int_min["min"]

    def _fit_resample(self, X: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
        X_resample, y_resample = self._to_numpy(X, y)

        X_resample, y_resample = self._fit(X_resample, y_resample)._resample(X_resample, y_resample)

        indices = np.arange(y_resample.shape[0])
        np.random.shuffle(indices)
        return X_resample[indices], y_resample[indices]

    def _fit(self, X: Any, y: Any) -> GMMS:
        self.class_sizes = Counter(y)
        self._construct_neighborhood(X, y)

        self._construct_maj_int_min()
        self._set_size_to_align()

        self._fit_each_minority_class(X, y)

        return self

    @staticmethod
    def _to_numpy(X: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(X).copy(), np.array(y).copy()

    def _fit_each_minority_class(self, X: np.ndarray, y: np.ndarray) -> None:
        for minority_class in self.minority_classes:
            self._fit_single_class(X, y, minority_class)

    def _fit_single_class(self, X: np.ndarray, y: np.ndarray, minority_class: int) -> None:
        self.__x_subset = X[y == minority_class]
        train: np.ndarray
        valid: np.ndarray
        train, valid = train_test_split(self.__x_subset, test_size=self.valid_size, random_state=self.random_state)

        current_component_count = self.min_components

        gaussian_mixture_model = self._init_model(current_component_count)
        gaussian_mixture_model_temp = None
        gaussian_mixture_model.fit(train)

        likelihood = [float("-inf"), gaussian_mixture_model.score(valid)]
        while self._perform_step(current_component_count, likelihood[1] - likelihood[0], train.shape[0]):
            if gaussian_mixture_model_temp is not None:
                gaussian_mixture_model = deepcopy(gaussian_mixture_model_temp)

            current_component_count += 1
            gaussian_mixture_model_temp = self._init_model(current_component_count)
            gaussian_mixture_model_temp.fit(train)

            likelihood[0], likelihood[1] = likelihood[1], gaussian_mixture_model_temp.score(valid)

        gaussian_mixture_model.fit(self.__x_subset)
        self.gaussian_mixtures[minority_class] = gaussian_mixture_model
        self.likelihoods[minority_class] = gaussian_mixture_model.score(self.__x_subset)
        self.__x_subset = None

    def _construct_neighborhood(self, X: np.ndarray, y: np.ndarray) -> None:
        neigh_clf: NearestNeighbors = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(X)
        nearest_neighbor_idxs: np.ndarray = neigh_clf.kneighbors(X, return_distance=False)[:, 1:]
        self.neighborhood = dict()
        neigh_samples: np.ndarray
        for sample_idx, neigh_samples in enumerate(nearest_neighbor_idxs):
            neigh_counts: Counter = Counter(y[neigh_samples])
            self.neighborhood[sample_idx] = self._check_sample_neighborhood(y[sample_idx], neigh_counts)

    def _check_sample_neighborhood(self, sample_class: int, neigh_counts: Counter[int]) -> float:
        neighborhood = 0.0
        for neigh_class, count in neigh_counts.items():
            class_sizes: List = [
                self.class_sizes[sample_class],
                self.class_sizes[neigh_class],
            ]
            neighborhood += count * (min(class_sizes) / max(class_sizes))
        neighborhood /= self.k_neighbors
        if neighborhood > 1:
            raise ValueError(f"Neighborhood is bigger than 1: {neighborhood}")
        return neighborhood

    def _construct_maj_int_min(self) -> None:
        middle_size = self._get_middle_size_based_on_strategy()
        self._fill_maj_int_min(middle_size)

    def _get_middle_size_based_on_strategy(self) -> int:
        if self.strategy == "median":
            middle_size = int(np.median(list(self.class_sizes.values())))
        elif self.strategy == "average":
            middle_size = np.mean(list(self.class_sizes.values()), dtype=int)
        else:
            raise ValueError(f'Unrecognized {self.strategy}. Only "median" and "average" are allowed.')
        return middle_size

    def _fill_maj_int_min(self, middle_size: int) -> None:
        for class_label, class_size in self.class_sizes.items():
            if class_size == middle_size:
                class_group = "int"
            elif class_size < middle_size:
                class_group = "min"
            else:
                class_group = "maj"

            self.maj_int_min[class_group].append(class_label)

    def _set_size_to_align(self) -> None:
        maj_q = [self.class_sizes[k] for k in self.maj_int_min["maj"]]
        min_q = [self.class_sizes[k] for k in self.maj_int_min["min"]]
        int_q = [self.class_sizes[k] for k in self.maj_int_min["int"]]

        if len(maj_q) == 0 and len(min_q) > 0:
            self.size_to_align = np.mean(min_q, dtype=int)
        elif len(min_q) == 0 and len(maj_q) > 0:
            self.size_to_align = np.mean(maj_q, dtype=int)
            return
        elif len(maj_q) > 0 and len(min_q) > 0:
            self.size_to_align = np.mean((max(min_q), min(maj_q)), dtype=int)
        elif len(int_q) > 0:
            self.size_to_align = np.mean(int_q, dtype=int)
        else:
            raise ValueError("Bad input - can not obtain desire size.")

    def _init_model(self, n_components: int) -> GaussianMixture:
        return GaussianMixture(
            n_components=n_components,
            n_init=self.n_init,
            covariance_type=self.covariance_type,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def _perform_step(self, n_components: int, likelihood: float, num_samples: int) -> bool:
        likelihood_condition = likelihood >= self.likelihood_threshold
        max_components_condition = self.max_components is None or n_components <= self.max_components
        num_samples_condition = n_components < num_samples
        return likelihood_condition and max_components_condition and num_samples_condition

    def _resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = self._oversample_each_minority_class(X, y)
        if self.undersample and "maj" in self.maj_int_min:
            X, y = self._undersample_majority_classes(X, y)
        return X, y

    def _oversample_each_minority_class(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_copy = X.copy()
        y_copy = y.copy()
        for minority_class in self.minority_classes:
            self.__x_subset = X_copy[y_copy == minority_class]
            X, y = self._oversample(X_copy, y_copy, minority_class)
            self.__x_subset = None
        return X, y

    def _oversample(self, X: np.ndarray, y: np.ndarray, minority_class: int) -> Tuple[np.ndarray, np.ndarray]:
        means, covariances = self._get_coefficients(self.gaussian_mixtures[minority_class])

        probabilities = self._get_probas_for_samples_in_component(X, y, minority_class)
        quantity_to_generate = self.size_to_align - self.__x_subset.shape[0]
        for component in range(self.gaussian_mixtures[minority_class].n_components):
            Nk: np.ndarray = probabilities[component] * quantity_to_generate
            x = self._create_samples(means[component], covariances[component], int(Nk))
            X = np.append(X, x, axis=0)
            y = np.append(y, np.full((x.shape[0],), fill_value=minority_class), axis=0)

        return X, y

    def _get_probas_for_samples_in_component(self, X: np.ndarray, y: np.ndarray, minority_class: int) -> np.ndarray:
        X_prob: np.ndarray = self.gaussian_mixtures[minority_class].predict_proba(X[y == minority_class])
        ratios = np.array([v for k, v in self.neighborhood.items() if y[k] == minority_class])
        ratios = ratios[..., np.newaxis]
        probabilities: np.ndarray = np.sum((1.0 - ratios) * X_prob, axis=0) + 1e-8
        probabilities = probabilities / np.sum(probabilities, keepdims=True)
        return probabilities

    def _get_coefficients(self, gaussian_mixture: GaussianMixture) -> Tuple[np.ndarray, np.ndarray]:
        means: np.ndarray = gaussian_mixture.means_
        covariances: np.ndarray = gaussian_mixture.covariances_
        if self.covariance_type == "tied":
            covariances = np.array([covariances] * gaussian_mixture.n_components)
        elif self.covariance_type == "diag":
            cov_list: List = []
            for component in range(gaussian_mixture.n_components):
                cov_list.append(np.diagflat(covariances[component, :]))
            covariances = np.array(cov_list)
        elif self.covariance_type == "spherical":
            cov_list: List = []
            for component in range(gaussian_mixture.n_components):
                var = np.array([covariances[component]] * self.__x_subset.shape[1])
                cov_list.append(np.diagflat(var))
            covariances = np.array(cov_list)
        return means, covariances

    def _create_samples(self, mean: np.ndarray, covariance: np.ndarray, target_size: int) -> np.ndarray:
        result = np.empty((0, self.__x_subset.shape[1]), float)
        iterations = 0
        threshold_dist = self.filter_new
        while (result.shape[0] != target_size) and (iterations < self.iterations_after_filtration):
            iterations += 1
            size = max(target_size - result.shape[0], result.shape[1] + 1)
            x = np.random.multivariate_normal(mean, covariance, size=size)
            if self.filter_new == -1.0:
                result = np.append(result, x, axis=0)
                break
            elif self.filter_new == 0.0:
                mdist = self._compute_mdist(self.__x_subset, mean, covariance)
                threshold_dist = float(np.mean(mdist))

            mdist = self._compute_mdist(x, mean, covariance)[: x.shape[0]]
            x = x[mdist < threshold_dist]
            x = x[: target_size - result.shape[0]]
            result = np.append(result, x, axis=0)
            if not self.add_after_filtration:
                break
        return result

    def _compute_mdist(self, in_data: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        try:
            data = in_data
            if data.shape[0] < self.cdist_min_count:
                data = np.concatenate((in_data, in_data), axis=0)
            mdist = cdist(data, [mean], metric="mahalanobis", VI=np.linalg.inv(covariance))[:, 0]
        except Exception as e:
            self.__logger.error("Can't compute 'cdist' function. Distance threshold is set to 2.0")
            self.__logger.info(f"For more information, examine exception: {e}")
            mdist = np.full_like(in_data, fill_value=2.0)[:, 0]
        return mdist

    def _undersample_majority_classes(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for maj_class in self.maj_int_min["maj"]:
            X, y = self._undersample(X, y, maj_class)
        return X, y

    def _undersample(self, X: np.ndarray, y: np.ndarray, class_id: int) -> Tuple[np.ndarray, np.ndarray]:
        class_idxs = np.where(y == class_id)[0]
        sorted_neigh = sorted(self.neighborhood.items(), key=lambda item: item[1])
        class_idxs = [k for k, _ in sorted_neigh if k in class_idxs]
        size = max(0, int(self.class_sizes[class_id] - self.size_to_align))
        X = np.delete(X, class_idxs[:size], axis=0)
        y = np.delete(y, class_idxs[:size], axis=0)

        return X, y
