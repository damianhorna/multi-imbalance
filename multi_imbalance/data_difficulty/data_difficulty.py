from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from matplotlib.figure import Figure
from typing import Optional, List, Dict, Tuple, TypeVar
from seaborn import heatmap


__author__ = 'Adam Wojciechowski'


class Summary:
    """A class to quickly run all of the implemented algorithms
    in form of a report with outputs in the form of various plots.
    """
    def fit(self, 
            X: NDArray, 
            y: NDArray, 
            classes: Optional[List[int]] = None, 
            k: int = 5, 
            n_jobs: int = -1) -> None:
        """Creates the summary report. 

        :param X: The examples
        :param y: The labels
        :param classes: List of classes to be processed, defaults to None
        :param k: No neighbors, defaults to 5
        :param n_jobs: No threads for distributed KNN action, defaults to -1
        """
        self.X, self.y = X, y
        self.classes = classes
        self.k, self.n_jobs = k, n_jobs
        print('Start computing imbalance ratios...')
        imbalance_ratio = ImbalanceRatio()
        self.imbalance_ratio_results = imbalance_ratio.fit(self.y)
        print('Imbalance ratios: ')
        for k, v in self.imbalance_ratio_results.items():
            print(f'class {k} : {v}')
        
        print('Start data difficulty study...')
        print('KNN based method')
        knn_data_difficulty = KNNDataDifficulty(classes=self.classes, 
                                                k=self.k, 
                                                n_jobs=self.n_jobs)
        self.knn_data_difficulty_results = knn_data_difficulty.fit_plot(self.X, 
                                                                        self.y)
        print('Kernel density function based method')
        kernel_data_difficulty = KernelDataDifficulty(classes=self.classes, 
                                                      k=self.k, 
                                                      n_jobs=self.n_jobs)
        self.kernel_data_difficulty_results = kernel_data_difficulty.fit_plot(self.X, 
                                                                              self.y)
        print('Continuous data difficulty')
        continuous_data_difficulty = ContinuousDataDifficulty(classes=self.classes, 
                                                              k=self.k, 
                                                              n_jobs=self.n_jobs)
        self.continuous_data_difficulty_results = continuous_data_difficulty.fit_plot(self.X, 
                                                                                      self.y)
                                                                                      
        print('Class neighboring study...')
        border_class_matrix = BorderClassMatrix(classes=self.classes, 
                                                k=self.k, 
                                                n_jobs=n_jobs)  
        self.border_class_matrix_results = border_class_matrix.fit_plot(self.X, 
                                                                        self.y)    
                                                                        
        print('Class homogeneity study...')
        print('KMeans Elbow')
        kmeans_elbow = KMeansElbowMethod(classes=self.classes, 
                                         cluster_count_selection=list(range(1, 15)))
        self.kmeans_elbow_results = kmeans_elbow.fit_plot(X=self.X, 
                                                          y=self.y)
        print('Gaussian mixture Elbow')
        gaussian_mixture = GaussianMixtureElbowMethod(classes=self.classes, 
                                                      cluster_count_selection=list(range(1, 15)))
        self.gaussian_mixture_results = gaussian_mixture.fit_plot(X=self.X, 
                                                                  y=self.y)
        
        
class ImbalanceRatio:
    """A class which performs imbalance ratio computations
    resulting in imbalance ratio values for each class.
    """
    def fit(self,
            y: NDArray) -> Dict[int, float]:
        """Main class method performing computations.

        :param y: The labels
        :return: Imbalance ratio values for each class in y
        """
        classes, cardinalities = np.unique(y, return_counts=True)
        maj_class_id = np.argmax(cardinalities)
        return {
            classes[i]: cardinalities[maj_class_id] / cardinalities[i]
            for i in range(len(classes))
        }
            

class DataDifficulty(ABC):
    """An abstract class serving as parent class to:
    
    * KNNDataDifficulty
    * ContinuousDataDifficulty
    * BorderClassMatrix
    * KernelDataDifficulty
    
    """    
    def __init__(self, 
                 classes: Optional[List[int]] = None, 
                 k: int = 5, 
                 n_jobs: int = -1) -> None:
        """Class constructor.

        :param classes: List of classes to be processed, defaults to None
        :param k: No neighbors, defaults to 5
        :param n_jobs: No threads for distributed KNN action, defaults to -1
        :raises TypeError: Raised if classes attribute is of invalid type,
            because the validity of this attribute is esential for all of the
            child classes
        """        
        if classes is not None and type(classes) is not list:
            raise TypeError('Wrong type of classes parameter! Should be None or list.')
        self.classes = classes
        self.k = k
        self.knn = NearestNeighbors(n_neighbors=self.k, n_jobs=n_jobs)
        
    @abstractmethod
    def fit(self, 
            X: NDArray, 
            y: NDArray) -> None:
        """Main class method performing the algorithm, to be implemented
        inside child classes.

        :param X: The examples
        :param y: The labels
        """        
        pass
            
    def _induce_classes(self, 
                        y: NDArray) -> NDArray:
        """Method for automatic minority class list induction
        if none was provided by the user.

        :param y: The labels
        :return: An array of minority class labels
        """        
        distribution = np.unique(y, return_counts=True)
        majority_cardinality = max(distribution[1])
        return distribution[0][distribution[1] < majority_cardinality]
    
    
class KernelFunction(ABC):
    """An abstract class serving as parent class to:
    
    * EpanechnikovKernelFunction
    * TriangularKernelFuntion
    * UniformKernelFunction

    """
    def __init__(self, kernel_bandwidth: float) -> None:
        """Class constructor.

        :param kernel_bandwidth: The bandwidth of the krenel function.
            The kernel function will return positive values, if 
            the argument `u` will be in range 
            (-kernel_bandwidth / 2; kernel_bandwidth / 2)
        """
        super().__init__()
        self.kernel_bandwidth = kernel_bandwidth
        self.scaling_factor = 2 / self.kernel_bandwidth
        
    @abstractmethod
    def K(self, u: float) -> float:
        """Kernel function.

        :param u: The argument of the kernel function K
        :return: The numerical value K(u)
        """
        pass
    
    
KernelFunctionSubclass = TypeVar('KernelFunctionSubclass', bound=KernelFunction)
    
    
class EpanechnikovKernelFunction(KernelFunction):
    """Epanechnikov (parabolic) kernel function.
    """
    def __init__(self, kernel_bandwidth: float) -> None:
        """Class constructor

        :param kernel_bandwidth: The bandwidth of the krenel function.
            The kernel function will return positive values, if 
            the argument `u` will be in range 
            (-kernel_bandwidth / 2; kernel_bandwidth / 2)
        """
        super().__init__(kernel_bandwidth)
        
    def K(self, u: float) -> float:
        """Kernel function.
        It is defined as: 
        K(u) = (3 / 4) * (1 - u ** 2), for |u| <= 1
        K(u) = 0, otherwise.

        :param u: The argument of the kernel function K
        :return: The numerical value K(u)
        """
        if np.abs(self.scaling_factor * u) <= 1:
            return (3 / 4) * (1 - np.square((self.scaling_factor * u)))
        else:
            return 0.
        
        
class TriangularKernelFuntion(KernelFunction):
    """Triangular (linear) kernel function.
    """
    def __init__(self, kernel_bandwidth: float) -> None:
        """Class constructor

        :param kernel_bandwidth: The bandwidth of the krenel function.
            The kernel function will return positive values, if 
            the argument `u` will be in range 
            (-kernel_bandwidth / 2; kernel_bandwidth / 2)
        """
        super().__init__(kernel_bandwidth)
        
    def K(self, u: float) -> float:
        """Kernel function.
        It is defined as: 
        K(u) = 1 - |u|, for |u| <= 1
        K(u) = 0, otherwise.

        :param u: The argument of the kernel function K
        :return: The numerical value K(u)
        """
        if np.abs(self.scaling_factor * u) <= 1:
            return 1 - np.abs(self.scaling_factor * u)
        else:
            return 0.
        
        
class UniformKernelFunction(KernelFunction):
    """Uniform (constant) kernel function.
    """
    def __init__(self, kernel_bandwidth: float) -> None:
        """Class constructor

        :param kernel_bandwidth: The bandwidth of the krenel function.
            The kernel function will return positive values, if 
            the argument `u` will be in range 
            (-kernel_bandwidth / 2; kernel_bandwidth / 2)
        """
        super().__init__(kernel_bandwidth)
        
    def K(self, u: float) -> float:
        """Kernel function.
        K(u) = 1 / 2, for |u| <= 1
        K(u) = 0, otherwise.

        :param u: The argument of the kernel function K
        :return: The numerical value K(u)
        """
        return 1 / 2 if np.abs(self.scaling_factor * u) <= 1 else 0.
        

class KernelDataDifficulty(DataDifficulty):
    """Kernel approach to data difficulty discovery.
    """
    def __init__(self, 
                 classes: Optional[List[int]] = None, 
                 k: int = 5, 
                 n_jobs: int = -1, 
                 kernel_type: str = 'epanechnikov') -> None:
        """Class constructor.

        :param classes: List of classes to be processed, defaults to None
        :param k: No neighbors, defaults to 5
        :param n_jobs: No threads for distributed KNN action, defaults to -1
        :param kernel_type: Type of kernel function to be used, defaults to 'epanechnikov'. 
            currently available kernel types:
            
            * 'epanechnikov'
            * 'triangular'
            * 'uniform'
            
        """
        super().__init__(classes, k, n_jobs)
        self.kernel_type = kernel_type
        
    def fit(self, 
            X: NDArray,
            y: NDArray,
            kernel_bandwidth_override: Optional[float] = None) -> Dict[int, List[float]]:
        """Method for running the algorithm.

        :param X: The examples
        :param y: The labels
        :param kernel_bandwidth_override: An optional float value to be set as kernel bandwidth.
            If not set the algorithm uses automatic bandwidth tuning method, defaults to None
        :return: Dictionary {class_label : [scores for the examples of a given class]}. 
            The ordering inside the lists is as if iterated over a minority class
        """
        if self.classes is None:
            self.classes = self._induce_classes(y)
        self.X, self.y = X, y
        if kernel_bandwidth_override is None:
            self.kernel_bandwidth = self._bandwidth_tune()
        else:
            self.kernel_bandwidth = kernel_bandwidth_override
        self.kernel_object = self._get_kernel_object(self.kernel_type, 
                                                     self.kernel_bandwidth)
        c_scores = {}
        for c in self.classes:
            c_scores[c] = []
            # x - currently considered from minority class
            for x in self.X[self.y==c]:
                # list of weighted distances between same (c) class examples
                c_weighted_distances = []
                # list of weighted distances between x and any other example
                any_weighted_distances = []
                # xx - every other from any class
                for xx, y_of_xx in zip(self.X, self.y):
                    if np.all(x == xx): continue
                    distance = np.linalg.norm(x - xx)
                    weighted_distance = self.kernel_object.K(distance)
                    any_weighted_distances.append(weighted_distance)
                    if y_of_xx in self.classes:     
                        c_weighted_distances.append(weighted_distance)
                c_weighted_sum = np.sum(c_weighted_distances)
                any_weighted_sum = np.sum(any_weighted_distances)
                # when conditions is met, according to authors
                # there is 'not enough info' to compute the score
                if any_weighted_sum == 0.:
                    c_scores[c].append(-1.)
                else:
                    c_scores[c].append(c_weighted_sum / any_weighted_sum)
            # round to eliminate numerical stability errors
            c_scores[c] = list(np.around(c_scores[c], decimals=10))
        return c_scores
    
    def fit_label_difficulty(self, 
            X: NDArray,
            y: NDArray,
            kernel_bandwidth_override: Optional[float] = None) -> Dict[int, List[str]]:
        """Performs self.fit() method, but translates scores to 
        safety levels (safe, border, rare, ...).

        :param X: The examples
        :param y: The labels
        :param kernel_bandwidth_override: An optional float value to be set as kernel bandwidth.
            If not set the algorithm uses automatic bandwidth tuning method, defaults to None
        :return: Dictionary {class_label : [safety levels for the examples of a given class]}. 
            The ordering inside the lists is as if iterated over a minority class
        """
        c_scores = self.fit(X, y, kernel_bandwidth_override)
        return {
            c: [self._switch_case(score) for score in scores]
            for c, scores in c_scores.items()
        }   
    
    def fit_plot(self, 
            X: NDArray, 
            y: NDArray,
            kernel_bandwidth_override: Optional[float] = None) -> List[Figure]:     
        """Performs self.fit() method, but presents its results
        as a histogram.

        :param X: The examples
        :param y: The labels
        :param kernel_bandwidth_override: An optional float value to be set as kernel bandwidth.
            If not set the algorithm uses automatic bandwidth tuning method, defaults to None
        :return: List of plots, each for every class specified in classes
        """
        c_scores = self.fit(X, y, kernel_bandwidth_override)
        plots = []
        for k, v in c_scores.items():
            fig = plt.figure()
            plt.title(f'Class {k}')
            plt.hist(v)
            plt.xlim((-1., 1.))
            plt.xlabel('Bins')
            plt.ylabel('Safe scores')
            plt.show()
            plots.append(fig)
        return plots
    
    def _switch_case(self, 
                     score: float) -> str:     
        """Helper, switch-case-alike method to map
        score to name.

        :param score: safe score
        :return: A safe level name
        """    
        if 1 >= score > .7:
            return 'safe'
        elif .7 >= score > .3:
            return 'border'
        elif .3 >= score > .1:
            return 'rare'
        elif .1 >= score > 0:
            return 'outlier'
        elif score == 0.:
            return 'zero'
        elif score == -1:
            return 'not enough info'
        else:
            raise ValueError('Score value out of spec!')
        
    def _bandwidth_tune(self) -> float:
        """A method for automatic kernel bandwidth tuning.

        :return: A numerical value being the kernel's bandwidth
            required to compute scaling factor in kernel object
        """
        self.knn.fit(self.X, self.y)
        kernel_bandwidth, counter_for_avg = 0., 0
        for c in self.classes:
            for x in self.X[self.y==c]:
                # return all k nearest distances
                distances, _ = self.knn.kneighbors([x], 
                                                   self.k + 1, 
                                                   return_distance=True)
                # only accumulate the kth distance
                kernel_bandwidth += distances.squeeze()[self.k]
                counter_for_avg += 1
        kernel_bandwidth /= counter_for_avg
        return kernel_bandwidth
    

    def _get_kernel_object(self, 
                           kernel_type: str, 
                           kernel_bandwidth: float) -> KernelFunctionSubclass:
        """Helper, switch-case-alike method to map krenel type (str)
        to an actual, appropriate kernel function object.

        :param kernel_type: name of the kernel function as specified in 
            class constructor
        :param kernel_bandwidth: A numerical value being kernel's bandwidth
            either computed automatically (self._bandwidth_tune() method) or 
            acquired with kernel_bandwidth_override attribute in self.fit() method
        :return: Kernel Function object respective to kernel_type
        """
        if kernel_type == 'epanechnikov':
            return EpanechnikovKernelFunction(kernel_bandwidth)
        elif kernel_type == 'triangular':
            return TriangularKernelFuntion(kernel_bandwidth)
        elif kernel_type == 'uniform':
            return UniformKernelFunction(kernel_bandwidth)
        

class KNNDataDifficulty(DataDifficulty):
    """Class which performs DataDifficulty algorithm,
    resulting in a dictionary. (see self._get_names_dict method)
    """    
    
    def __init__(self, 
                 classes: Optional[List[int]] = None, 
                 k: int=5, 
                 n_jobs: int=-1) -> None:
        """Class constructor.

        :param classes: List of classes to be processed, defaults to None
        :param k: No neighbors, defaults to 5
        :param n_jobs: No threads for distributed KNN action, defaults to -1
        """  
        super().__init__(classes, k, n_jobs)
        
    def fit(self, 
            X: NDArray,
            y: NDArray) -> Dict[int, Dict[str, int]]:
        """Method for running the algorithm.

        :param X: The examples
        :param y: The labels
        :return: Dictionary {class_label : {safe_level_name : count}}
        """        
        if self.classes is None:
            self.classes = self._induce_classes(y)
        self.X, self.y = X, y
        self.knn.fit(self.X, self.y)
        non_c_count = {}
        for c in self.classes:
            non_c_count[c] = self._get_names_dict()
            for x in self.X[self.y==c]:
                neighbors = self.knn.kneighbors([x], self.k + 1, return_distance=False)
                neighbors = np.squeeze(neighbors)[1:]
                neighbors = self.y[neighbors]
                count_non_c = sum(neighbors != c)
                non_c_count[c][self._switch_case(count_non_c)] += 1
        return non_c_count
    
    def fit_plot(self, 
            X: NDArray, 
            y: NDArray) -> List[Figure]:
        """Performs fit method and returns its results as histograms.

        :param X: The examples
        :param y: The labels
        :return: List of plots, each for every class specified in classes
        """        
        fit_dict = self.fit(X, y)
        plots = []
        for k, v in fit_dict.items():
            fig = plt.figure()
            plt.title(f'Class {k}')
            plt.xticks(rotation=90)
            plt.bar(v.keys(), v.values())
            plt.xlabel('Type')
            plt.ylabel('Count')
            plt.show()
            plots.append(fig)
        return plots
                
    def _get_names_dict(self) -> Dict[str, int]:
        """Method for safe level dictionary creation.

        :return: safe level dictionary
        """        
        return {'safe': 0, 'border': 0, 'rare': 0, 'outlier': 0}
    
    def _switch_case(self, 
                     count: int) -> str:
        """Helper, switch-case-alike method to map
        count to name.

        :param count: Non-c neighbor count
        :return: A safe level name
        """        
        if count < 2:
            return 'safe'
        elif count < 4:
            return 'border'
        elif count < 5:
            return 'rare'
        else:
            return 'outlier'
        

class ContinuousDataDifficulty(DataDifficulty):
    """Class which performs DataDifficulty algorithm,
    resulting in a histogram. (see self.fit_plot method)
    """    
    
    def __init__(self, 
                 classes: Optional[List[int]] = None, 
                 k: int = 5, 
                 n_jobs: int = -1) -> None:
        """Class constructor.

        :param classes: List of classes to be processed, defaults to None
        :param k: No neighbors, defaults to 5
        :param n_jobs: No threads for distributed KNN action, defaults to -1
        """  
        super().__init__(classes, k, n_jobs)
    
    def fit(self, 
            X: NDArray, 
            y: NDArray, 
            similarities: Optional[NDArray] = None) -> Dict[int, List[float]]:
        """Method for running the algorithm.

        :param X: The examples
        :param y: The labels
        :param similarities: Similarity matrix, defaults to None
        :raises ValueError: When user passed a non-minority class to
            classes inside the constructor
        :return: Dictionary {class_label : [scores for the examples of a given class]}. 
            The ordering inside the lists is as if iterated over a minority class
        """        
        induced_classes = self._induce_classes(y)
        if self.classes is None:
            self.classes = induced_classes
        elif not set(self.classes).issubset(set(induced_classes)):
            error_msg = '''Cannot perform this method on non-minority classes! 
            (majority class has been passed to classes)'''
            raise ValueError(error_msg)
        if similarities is None:
            all_classes = np.unique(y)
            self.similarities = np.zeros(shape=(all_classes.shape[0], 
                                                all_classes.shape[0]), 
                                         dtype=np.int64)
            # self.classes are the classes to be processed
            for c in self.classes:
                # all_classes is a set of every possible class in the dataset
                for cc in all_classes:
                    if c == cc: continue
                    # if current cc (any kind of class) is a minority class
                    if cc in self.classes:
                        self.similarities[c, cc] = 1
        else:
            self.similarities = similarities
        self.X, self.y = X, y
        self.knn.fit(self.X, self.y)
        c_scores = {}
        for c in self.classes:
            c_scores[c] = []
            for x in self.X[self.y==c]:
                neighbors = self.knn.kneighbors([x], self.k + 1, return_distance=False)
                neighbors = np.squeeze(neighbors)[1:]
                neighbors = self.y[neighbors]
                non_c_neighbors = neighbors[neighbors != c]
                # if no non-c neighbors, the score is zero
                if not np.size(non_c_neighbors):
                    c_scores[c].append(0.)
                # else, the score is calculated according to the formula
                else:
                    safe_score = [self.similarities[c,neighbor] for neighbor in non_c_neighbors]
                    safe_score = sum(safe_score)
                    safe_score /= self.k
                    c_scores[c].append(safe_score)
        return c_scores
        
    def fit_plot(self, 
            X: NDArray, 
            y: NDArray, 
            similarities: Optional[NDArray] = None) -> List[Figure]:
        """Performs fit method and returns its results as histograms.

        :param X: The examples
        :param y: The labels
        :param similarities: Similarity matrix, defaults to None
        :return: List of plots, each for every class specified in classes
        """        
        c_scores = self.fit(X, y, similarities)
        plots = []
        for k, v in c_scores.items():
            fig = plt.figure()
            plt.title(f'Class {k}')
            plt.hist(v)
            plt.xlim((0., 1.))
            plt.xlabel('Bins')
            plt.ylabel('Safe scores')
            plt.show()
            plots.append(fig)
        return plots
        

class BorderClassMatrix(DataDifficulty):
    """Class which performs Border Class Matrix algorithm,
    resulting in a border class matrix.
    """    
    
    def __init__(self, 
                 classes: Optional[List[int]] = None, 
                 k: int = 5, 
                 n_jobs: int = -1) -> None:
        """Class constructor.

        :param classes: List of classes to be processed, defaults to None
        :param k: No neighbors, defaults to 5
        :param n_jobs: No threads for distributed KNN action, defaults to -1
        """  
        super().__init__(classes, k, n_jobs)
        
    def fit(self, 
            X: NDArray, 
            y: NDArray) -> NDArray:
        """Runs the algorithm.

        :param X: The examples
        :param y: The labels
        :return: Border class matrix
        """        
        if self.classes is None:
            self.classes = self._induce_classes(y)
        self.X, self.y = X, y
        all_classes_count = int(np.max(y)) + 1
        neighbor_matrix = np.zeros((all_classes_count, 
                                    all_classes_count), 
                                   dtype=np.int64)
        self.knn.fit(self.X, self.y)
        for c in self.classes:
            for x in self.X[self.y==c]:
                neighbors = self.knn.kneighbors([x], self.k + 1, return_distance=False)
                neighbors = np.squeeze(neighbors)[1:]
                neighbors = self.y[neighbors]
                non_c_neighbors = neighbors[neighbors != c]
                if not np.size(non_c_neighbors): continue
                for neighboring_class in non_c_neighbors:
                    neighbor_matrix[c, neighboring_class] += 1
        return neighbor_matrix
    
    def fit_plot(self, 
                 X: NDArray, 
                 y: NDArray) -> Figure:     
        matrix = self.fit(X, y)
        fig = heatmap(data=matrix, 
                      annot=True, 
                      cmap='jet')
        plt.title('Per class crossovers')
        plt.show()
        return fig
        

class WCSSElbowMethod(ABC):        
    """An abstract class serving as parent class to:
    
    * KMeansElbowMethod
    * GaussianMixtureElbowMethod.
    
    This class computes within-cluster sum of squares
    for each cluster, created by underlying clustering method,
    and produces an elbow plot. Upon the analysis of this plot
    user can assess how many clusters a given class comprises of.
    """        
    def __init__(self,
                 classes: List[int], 
                 cluster_count_selection: List[int]) -> None:
        """Class constructor.

        :param classes: List of classes to be processed, defaults to None
        :param cluster_count_selection: List of cluster counts to be put on 
            x axis of the elbow plot
        """        
        super().__init__()
        self.classes = classes
        self.cluster_count_selection = cluster_count_selection
        
    def _induce_classes(self, 
                        y: NDArray) -> NDArray:
        """Method for automatic minority class list induction
        if none was provided by the user.

        :param y: The labels
        :return: An array of minority class labels
        """        
        distribution = np.unique(y, return_counts=True)
        majority_cardinality = max(distribution[1])
        return distribution[0][distribution[1] < majority_cardinality]
        
    @abstractmethod
    def _cluster(self, 
                 n_clusters: int) -> NDArray:
        """Performs clustering, which would in n_clusters clusters,
        with some clustering algorithm, to be implemented inside child classes.

        :param n_clusters: No clusters for the clustering method
        :return: An array of new labels (not y) artificially created by 
            clustering algorithm
        """        
        pass
    
    def _wcss(self, 
              X: NDArray, 
              clustering_induced_labels: NDArray) -> float:
        """Computes within-cluster sum of squares for all clusters, 
        for a given cluster labeling.

        :param X: The examples
        :param clustering_induced_labels: cluster-induced labels
        :return: WCSS value
        """        
        meta_sum = 0.
        for label in np.unique(clustering_induced_labels):
            X_with_label = X[clustering_induced_labels == label]
            mean = np.mean(X_with_label, axis=0)
            sum_of_squares = 0.
            for data_point in X_with_label:
                sum_of_squares += np.square(data_point - mean)
            meta_sum += sum_of_squares
        return meta_sum.sum().item()
    
    def fit(self, 
            X: NDArray, 
            y: NDArray) -> Dict[int, List[Tuple[int, float]]]:
        """Runs the algorithm, resulting in a dictionary.

        :param X: The examples
        :param y: The labels
        :return: A dictionary {class_name : [(cluster_count, wcss for the given cluster_count), ...]}
        """  
        self.X, self.y = X, y
        if self.classes is None:
            self.classes = self._induce_classes(y)      
        wcss_dict = {}
        for c in self.classes:
            X_c = self.X[self.y == c]
            wcss_dict[c] = []
            for cluster_count in self.cluster_count_selection:
                clustering_induced_labels = self._cluster(X=X_c, 
                                                          n_clusters=cluster_count)
                # results for a single cluster
                wcss_results = self._wcss(X_c, clustering_induced_labels)
                # add to the results of all clusters
                wcss_dict[c].append((cluster_count, wcss_results))
        return wcss_dict
    
    def fit_plot(self, 
                X: NDArray, 
                y: NDArray) -> List[Figure]:
        """
        Performs fit method and returns its results as plots.
        :return: List of plots, each for every class specified in classes
        """        
        wcss_dict = self.fit(X, y)
        plots = []
        for k, v in wcss_dict.items():
            fig = plt.figure()
            plt.title(f'Class {k}')
            x, y = list(zip(*v))
            plt.plot(x, y)
            plt.xticks(x)
            plt.xlabel('Clusters')
            plt.ylabel('WCSS')
            plt.show()
            plots.append(fig)
        return plots
                   
            
class KMeansElbowMethod(WCSSElbowMethod):
    """Performs elbow method with KMeans as underlying
    clustering algorithm.
    """    
    def __init__(self,
                 classes: List[int], 
                 cluster_count_selection: List[int]) -> None:    
        """Class constructor.

        :param classes: List of classes to be processed, defaults to None
        :param cluster_count_selection: List of cluster counts to be put on 
            x axis of the elbow plot
        """    
        super().__init__(classes, cluster_count_selection)
        
    def _cluster(self, 
                 X: NDArray, 
                 n_clusters: int) -> NDArray:
        """Performs clustering with KMeans.

        :param X: The examples
        :param n_clusters: Desired cluster count
        :return: An array of clustering induced labels
        """        
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        return kmeans.labels_
    
    
class GaussianMixtureElbowMethod(WCSSElbowMethod):
    """Performs elbow method with GaussianMixture(Expectation/Maximization)
    as underlying clustering algorithm.
    """  
    def __init__(self,
                 classes: List[int], 
                 cluster_count_selection: List[int]) -> None:
        """Class constructor.

        :param classes: List of classes to be processed, defaults to None
        :param cluster_count_selection: List of cluster counts to be put on 
            x axis of the elbow plot
        """   
        super().__init__(classes, cluster_count_selection)
        
    def _cluster(self, 
                 X: NDArray, 
                 n_clusters: int) -> NDArray:
        """Performs clustering with GaussianMixture(Expectation/Maximization).

        :param X: The examples
        :param n_clusters: Desired cluster count
        :return: An array of clustering induced labels
        """  
        gm = GaussianMixture(n_components=n_clusters)
        return gm.fit_predict(X)
