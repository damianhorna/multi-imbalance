"""Implementations of SMOM tqchnique and DBOS clustering algorithm."""

from collections import Counter, defaultdict

import numpy as np

from typing import List, Dict, Optional, Sequence

import sklearn.utils
from sklearn import neighbors
from multi_imbalance.utils import array_util
from imblearn.base import BaseSampler


def _nbdos(Sc: List[int], k: int,
           sNk: Dict[int, List[int]], rTh: float,
           nTh: int):
    """
    NBDOS clustering algorithm implementation.

    Reference:
    Zhu, Tuanfei, Yaping Lin, and Yonghe Liu. "Synthetic minority oversampling
    technique for multiclass imbalance problems." Pattern Recognition 72
    (2017): 327-340.

    :param Sc:
        Indices of items to perform clustering on.
    :param k:
        The number of nearest neighbors.
    :param sNk:
        The k-nearest neighbors lists.
    :param rTh:
        The minimal proportion of c class instances which should be
        achieved for the soft core instances in their k-nearest neighbors.
    :param nTh:
        The minimal number of members required for the discovered clusters.
    """

    # Naming of variables follows the pseudo code of nbdos
    xcl = defaultdict(int)
    hck = dict()
    sfSc = set()

    def _expand_cluster(j, cur_id):
        sfC = {j}
        xcl[j] = cur_id
        while sfC:
            j = next(iter(sfC))
            for L in hck[j]:
                if xcl[L] == 0:
                    xcl[L] = cur_id
                    if L in sfSc:
                        sfC.add(L)
            sfC.remove(j)

    for i in Sc:
        Tem = sNk[i]
        if round(Tem.size / k) >= rTh:
            sfSc.add(i)
            hck[i] = set(Tem)

    for i in sfSc:
        Tem = set((j for j, tem in hck.items() if i in tem))
        hck[i].update(Tem & sfSc)

    curId = 0
    for i in sfSc:
        if xcl[i] == 0:
            curId += 1
            _expand_cluster(i, curId)
    for i in range(1, curId + 1):
        ci = [j for j, cluster in xcl.items() if cluster == i]
        if len(ci) < nTh:
            for j in ci:
                xcl[j] = 0
    Scl = np.array([xcl[i] for i in Sc])
    return Scl


def _compute_ss(Fs_i, Fs_d, i, dst):
    Ss = []
    for idx, dst_ in zip(Fs_i[i], Fs_d[i]):
        if dst_ <= dst:
            Ss.append(idx)
    return np.array(Ss, dtype=np.int64)


def _normalized_entropy(classes_counts: Sequence[int]):
    # TODO: compare with the pseudo code in the article (falbogowski)
    if len(classes_counts) <= 1:
        E = 0
    else:
        total = sum(classes_counts)
        E_min = np.log(1. / total)
        E = sum(count / total * np.log(count / total) for count in
                classes_counts) / E_min
    assert 0 <= E <= 1, f'{E=} not in range [0, 1]'
    return E


class SMOM(BaseSampler):
    """
    SMOM technique implementation for synthetic minority oversampling for multiclass imbalanced problems.

    Reference:
    Zhu, Tuanfei, Yaping Lin, and Yonghe Liu. "Synthetic minority oversampling
    technique for multiclass imbalance problems." Pattern Recognition 72
    (2017): 327-340.
    """

    def __init__(self,
                 c: int,
                 zeta: int,
                 k1: int = 12,
                 k2: int = 8,
                 rTh: float = 5 / 8,
                 nTh: int = 10,
                 w1: float = 0.2,
                 w2: float = 0.5,
                 r1: float = 1 / 3,
                 r2: float = 0.2,
                 maj_int_min: Optional[Dict[str, Sequence[int]]] = None,
                 shuffle: bool = False,
                 metric: str = 'minkowski',
                 p: int = 2,
                 seed: Optional[int] = None) -> None:
        """
        :param c:
            The minority class under consideration.
        :param zeta:
            Number of synthetic instances to be generated.
        :param k1:
            Number of nearest neighbors used to generate the synthetic instances.
        :params k2, rTh, nTh:
            The parameters used in clustering algorithm NBDOS.
        :params w1, w2, r1, r2:
            The parameters used for calculating the selection weights.
        :param maj_int_min:
            Dict that contains lists of majority, intermediate and minority classes labels.
        :param shuffle:
            Shuffle resampled data.
        :param metric:
            Metric to use for distance computation.
        :param p:
            Power parameter for Minkowski metric.
        :param seed:
            Seed for random state.
        """
        super().__init__()
        self._sampling_type = 'over-sampling'
        self.maj_int_min = maj_int_min
        self.c = c
        self.zeta = zeta
        self.k1 = k1
        self.k2 = k2
        self.k3 = max(k1, k2)
        self.rTh = rTh
        self.nTh = nTh
        self.w1 = w1
        self.w2 = w2
        self.r1 = r1
        self.r2 = r2
        self.shuffle = shuffle
        self.random_state = sklearn.utils.check_random_state(seed)
        if metric == 'minkowski':
            self._metric = neighbors.DistanceMetric.get_metric(metric, p=p)
        else:
            self._metric = neighbors.metrics.DistanceMetric.get_metric(metric)

    def _pairwise_distance(self, x1, x2):
        return self._metric.pairwise([x1], [x2])[0, 0]

    def _dpn(self, X, i, j, Ss, dis_ij):
        # Naming of variables follows the pseudo code of dPN
        N_pn = [i, j]
        mid_ij = (X[i] + X[j]) * 0.5
        for L in Ss:
            dis_lm = self._pairwise_distance(X[L], mid_ij)
            if 2 * dis_lm <= dis_ij:
                N_pn.append(L)
        return N_pn

    def _find_nearest_k3_in_sc(self, X, Sc, i):
        dist_, ind_ = self.kdt_c.query([X[i]], self.k3 + 1)
        ind, dist = ind_[0][1:], dist_[0][1:]
        ind = Sc[ind]
        self.N_c_k3_i[i] = ind
        self.N_c_k3_d[i] = dist
        self.N_c_k1_i[i] = ind[:self.k1]
        self.N_c_k1_d[i] = dist[:self.k1]
        self.r_k1_i[i] = ind[self.k1 - 1]
        self.r_k1_d[i] = dist[self.k1 - 1]

    def _find_nearest_k3_in_sct(self, X, Sct, i):
        dist_, ind_ = self.kdt_ct.query([X[i]], self.k3)
        ind, dist = ind_[0], dist_[0]
        ind = Sct[ind]
        self.N_ct_k3_i[i] = ind
        self.N_ct_k3_d[i] = dist
        T_ind, T_dist = [], []
        for t_ind, t_dist in zip(ind, dist):
            if t_dist <= self.r_k1_d[i]:
                T_ind.append(t_ind)
                T_dist.append(t_dist)
        self.Fs_i[i] = np.concatenate([T_ind, self.N_c_k1_i[i]], axis=0)
        self.Fs_d[i] = np.concatenate([T_dist, self.N_c_k1_d[i]], axis=0)

    def _find_k2_nearest_in_neighbor(self, X, i):
        nc_nct_union_i = np.concatenate([self.N_c_k3_i[i], self.N_ct_k3_i[i]],
                                        axis=0)
        nc_nct_union_d = np.concatenate([self.N_c_k3_d[i], self.N_ct_k3_d[i]],
                                        axis=0)
        dist_, ind_ = neighbors.KDTree(X[nc_nct_union_i],
                                       metric=self._metric).query([X[i]], self.k2)
        ind, dist = ind_[0], dist_[0]
        self.N_k2_i[i] = nc_nct_union_i[ind]
        self.N_k2_d[i] = nc_nct_union_d[ind]

    def _compute_min_maj(self, y):
        if self.maj_int_min is None:
            M = y.size
            L = len(set(y))
            cnt = Counter(y[y != self.c])
            y_maj_classes = {k: v for k, v in cnt.items() if v >= M / L}
            y_min_classes = {k: v for k, v in cnt.items() if v < M / L}
        else:
            cnt = Counter(y)
            if 'int' not in self.maj_int_min:
                self.maj_int_min['int'] = []
            y_maj_classes = {cls: cnt[cls] for cls in
                             self.maj_int_min['maj'] + self.maj_int_min['int']}
            y_min_classes = {cls: cnt[cls] for cls in self.maj_int_min['min']}
        return y_maj_classes, y_min_classes

    def _run_nbdos(self, Sc):
        N_k2_dct = {index: neighbors_ for index, neighbors_ in
                    self.N_k2_i.items()}
        return _nbdos(Sc, self.k2, N_k2_dct, self.rTh, self.nTh)

    def _compute_selection_weight(self, X, Sc, i, j, y_min_classes,
                                  y_maj_classes):
        dis_ij = self._pairwise_distance(X[i], X[j])
        Ss = _compute_ss(self.Fs_i, self.Fs_d, i, dis_ij)
        N_pn = self._dpn(X, i, j, Ss, dis_ij)
        y_mi_pn = {k: v for k, v in y_min_classes.items() if
                   k in N_pn}
        y_ma_pn = {k: v for k, v in y_maj_classes.items() if
                   k in N_pn}
        y_mi = sum(y_mi_pn.values())
        y_ma = sum(y_ma_pn.values())
        E_mi = _normalized_entropy(y_mi_pn.values())
        E_ma = _normalized_entropy(y_ma_pn.values())

        y_c = len(Sc)
        exponent1 = self.r1 * y_mi / y_c + self.r2 * E_mi + self.w2 * (
                self.r1 * y_ma / y_c + self.r2 * E_ma)
        exponent2 = -y_c / (y_ma + y_mi + y_c)
        return 1.0 / np.exp(exponent1) + self.w1 * np.exp(
            exponent2)

    def _compute_selection_weights(self, X, Sc, TiC, OiC, y_min_classes,
                                   y_maj_classes):
        Sw = defaultdict(dict)
        for i in TiC:
            for j in self.N_c_k1_i[i]:
                if j in OiC and j in self.N_c_k1_i and i in self.N_c_k1_i[j]:
                    Sw[i][j] = 1 + self.w1 / np.e
                elif j in self.N_c_k1_i and i in self.N_c_k1_i[
                    j] and j in Sw and i in \
                        Sw[j]:
                    Sw[i][j] = Sw[j][i]
                else:
                    Sw[i][j] = self._compute_selection_weight(X, Sc, i, j,
                                                              y_min_classes,
                                                              y_maj_classes)
        return Sw

    def _obtain_probability_distribution(self, X, y, Sw, TiC):
        P = defaultdict(dict)
        for i in TiC:
            for j in Sw[i]:
                dis_ij = self._pairwise_distance(X[i], X[j])
                Ss = _compute_ss(self.Fs_i, self.Fs_d, i, dis_ij)
                PN = self._dpn(X, i, j, Ss, dis_ij)
                if (y[PN] == self.c).all():
                    break
            else:
                if i not in self.N_c_k1_i[i]:
                    self.N_c_k1_i[i] = np.append(self.N_c_k1_i[i], i)
                    self.N_c_k1_d[i] = np.append(self.N_c_k1_d[i], 0.)
                Sw[i][i] = 1 + self.w1 / np.e
            for j in Sw[i]:
                P[i][j] = Sw[i][j] / sum(Sw[i][k] for k in self.N_c_k1_i[i])
        return P

    def _compute_number_of_synthetic_instances(self, Sc):
        N_syn = dict()
        div, mod = divmod(self.zeta, Sc.shape[0])
        for i in Sc:
            N_syn[i] = div + (mod > 0)
            mod -= 1
        return N_syn

    def _generate_synthetic_instances(self, X, Sc, N_syn, TiC, P):
        SI = []
        for i in Sc:
            for _ in range(N_syn[i]):
                if i in TiC:
                    p = [P[i][j] for j in self.N_c_k1_i[i]]
                    j = self.random_state.choice(self.N_c_k1_i[i], size=1, p=p)
                else:
                    j = self.random_state.choice(self.N_c_k1_i[i], size=1)
                si = X[i] + (X[j] - X[i]) * self.random_state.rand(*X[j].shape)
                SI.append(si)
        return np.concatenate(SI, 0)

    def _setup(self):
        # nearest k3 instances from x_i in Sc
        self.N_c_k3_i = dict()
        self.N_c_k3_d = dict()
        # nearest k1 instances from x_i in Sc
        self.N_c_k1_i = dict()
        self.N_c_k1_d = dict()
        # nearest k3 instances from x_i in Sct
        self.N_ct_k3_i = dict()
        self.N_ct_k3_d = dict()
        # nearest k2 instances from x_i in union of N_c_k3[i] and N_ct_k3[i]
        self.N_k2_i = dict()
        self.N_k2_d = dict()

        self.r_k1_d = dict()
        self.r_k1_i = dict()

        self.Fs_i = dict()
        self.Fs_d = dict()

    def _fit_resample(self, X, y):
        """
        Performs resampling

        :param X:
            Numpy array of examples that is the subject of resampling.
        :param y:
            Numpy array of labels corresponding to examples from X.
        :return:
            Resampled X along with accordingly modified labels, resampled y
        """

        # 1
        Sc = np.array([i for i, _ in enumerate(X) if y[i] == self.c])
        Sct = np.array([i for i, _ in enumerate(X) if y[i] != self.c])

        y_maj_classes, y_min_classes = self._compute_min_maj(y)

        self._setup()

        self.kdt_c = neighbors.KDTree(X[Sc], metric=self._metric)
        self.kdt_ct = neighbors.KDTree(X[Sct], metric=self._metric)

        for i in Sc:
            self._find_nearest_k3_in_sc(X, Sc, i)
            self._find_nearest_k3_in_sct(X, Sct, i)
            self._find_k2_nearest_in_neighbor(X, i)

        Sc_cl = self._run_nbdos(Sc)

        OiC = Sc[Sc_cl != 0]
        TiC = Sc[Sc_cl == 0]

        Sw = self._compute_selection_weights(X, Sc, TiC, OiC, y_min_classes,
                                             y_maj_classes)
        P = self._obtain_probability_distribution(X, y, Sw, TiC)
        N_syn = self._compute_number_of_synthetic_instances(Sc)
        SI = self._generate_synthetic_instances(X, Sc, N_syn, TiC, P)

        X_resampled = np.concatenate([X, SI], 0)
        y_resampled = np.concatenate([y, [self.c] * SI.shape[0]], 0)

        if self.shuffle:
            X_resampled, y_resampled = array_util.shuffle(X_resampled,
                                                          y_resampled,
                                                          state=self.random_state)
        return X_resampled, y_resampled
