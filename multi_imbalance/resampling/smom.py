from collections import Counter, defaultdict

import numpy as np

from typing import List, Dict, Optional, Sequence

from sklearn import neighbors
from multi_imbalance.utils import array_util
from imblearn.base import BaseSampler


def _expand_cluster(self, sfSc, i, curId):
    sfC = {i}
    self._xcl[i] = curId
    while sfC:
        j = next(iter(sfC))
        for L in self._hck[j]:
            self._xcl[L] = curId
            if L in sfSc:
                sfC.add(L)
        sfC.remove(j)

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
        Indices of items to perform clustering on
    :param k:
        The number of nearest neighbors
    :param sNk:
        The k-nearest neighbors lists
    :param rTh:
        The minimal proportion of c class instances which should be
        achieved for the soft core instances in their k-nearest neighbors
    :param nTh:
        The minimal number of members required for the discovered clusters
    """
    xcl = defaultdict(int)
    hck = dict()
    sfSc = set()

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
            _expand_cluster(sfSc, i, curId)
    for i in range(1, curId + 1):
        ci = [j for j, cluster in xcl.items() if cluster == i]
        if len(ci) < nTh:
            for j in ci:
                xcl[j] = 0
    Scl = np.array([xcl[i] for i in Sc])
    return Scl




def _dpn(X, i, j, Ss):
    N_pn = [i, j]
    mid_ij = (X[i] + X[j]) * 0.5
    dis_ij = np.linalg.norm(X[i] - X[j])
    for l in Ss:
        dis_lm = np.linalg.norm(X[l] - mid_ij)
        if 2 * dis_lm <= dis_ij:
            N_pn.append(l)
    return N_pn


def _compute_ss(Fs_i, Fs_d, i, dst):
    Ss = []
    for i_, d_ in zip(Fs_i[i], Fs_d[i]):
        if d_ <= dst:
            Ss.append(i_)
    return np.array(Ss, dtype=np.int64)


def _normalized_entropy(classes_counts: Sequence[int]):
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

    def __init__(self, c: int, zeta: int, k1: int = 12, k2: int = 8,
                 rTh: float = 5 / 8,
                 nTh: int = 10, w1: float = 0.2, w2: float = 0.5,
                 r1: float = 1 / 3, r2: float = 0.2,
                 maj_int_min: Optional[Dict[str, List[int]]] = None,
                 shuffle: bool = False,
                 metric: str = 'minkowski',
                 p: int = 2) -> None:
        """
        :param maj_int_min:
            dict {'maj': majority class labels, 'min': minority class labels}
        :param c:
            The minority class under consideration
        :param zeta:
            Number of synthetic instances to be generated
        :param k1:
            Number of nearest neighbors used to generate the synthetic instances
        :params k2, rTh, nTh:
            The parameters used in clustering algorithm NBDOS
        :params w1, w2, r1, r2:
            The parameters used for calculating the selection weights
        :param maj_int_min:
            Dict that contains lists of majority, intermediate and minority classes labels.
        :param shuffle:
            Shuffle resampled data
        :param metric:
            Metric to use for distance computation.
        :param p:
            Power parameter for Minkowski metric.
        """
        super().__init__()
        self._sampling_type = 'over-sampling'
        self.maj_int_min = maj_int_min
        self.c = c
        self.zeta = zeta
        self.k1 = k1
        self.k2 = k2
        self.rTh = rTh
        self.nTh = nTh
        self.w1 = w1
        self.w2 = w2
        self.r1 = r1
        self.r2 = r2
        self.shuffle = shuffle
        if metric == 'minkowski':
            self._metric = neighbors.DistanceMetric.get_metric(metric, p=p)
        else:
            self._metric = neighbors.metrics.DistanceMetric.get_metric(metric)

    def _pairwise_distance(self, x1, x2):
        return self._metric.pairwise([x1], [x2])[0, 0]

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
        self.k1, self.k2, self.r1, self.r2
        k3 = max(self.k1, self.k2)

        # 1
        Sc = np.array([i for i, _ in enumerate(X) if y[i] == self.c])
        Sct = np.array([i for i, _ in enumerate(X) if y[i] != self.c])

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

        # nearest k3 instances from x_i in Sc
        N_c_k3_i = dict()
        N_c_k3_d = dict()
        # nearest k1 instances from x_i in Sc
        N_c_k1_i = dict()
        N_c_k1_d = dict()
        # nearest k3 instances from x_i in Sct
        N_ct_k3_i = dict()
        N_ct_k3_d = dict()
        # nearest k2 instances from x_i in union of N_c_k3[i] and N_ct_k3[i]
        N_k2_i = dict()
        N_k2_d = dict()

        r_k1_d = dict()
        r_k1_i = dict()

        Fs_i = dict()
        Fs_d = dict()

        kdt_c = neighbors.KDTree(X[Sc])
        kdt_ct = neighbors.KDTree(X[Sct])
        for i in Sc:
            # 2: 1)
            dist_, ind_ = kdt_c.query([X[i]], k3 + 1)
            ind, dist = ind_[0][1:], dist_[0][1:]
            N_c_k3_i[i] = ind
            N_c_k3_d[i] = dist
            N_c_k1_i[i] = ind[:self.k1]
            N_c_k1_d[i] = dist[:self.k1]
            r_k1_i[i] = ind[self.k1 - 1]
            r_k1_d[i] = dist[self.k1 - 1]

            # 2: 2)
            dist_, ind_ = kdt_ct.query([X[i]], k3)
            ind, dist = ind_[0], dist_[0]
            N_ct_k3_i[i] = ind
            N_ct_k3_d[i] = dist
            T_ind, T_dist = [], []
            for t_ind, t_dist in zip(ind, dist):
                if t_dist <= r_k1_d[i]:
                    T_ind.append(t_ind)
                    T_dist.append(t_dist)
            Fs_i[i] = np.concatenate([T_ind, N_c_k1_i[i]], axis=0)
            Fs_d[i] = np.concatenate([T_dist, N_c_k1_d[i]], axis=0)

            # 2: 3)
            nc_nct_union_i = np.concatenate([N_c_k3_i[i], N_ct_k3_i[i]],
                                            axis=0)
            nc_nct_union_d = np.concatenate([N_c_k3_d[i], N_ct_k3_d[i]],
                                            axis=0)
            dist_, ind_ = neighbors.KDTree(X[nc_nct_union_i]).query([X[i]])
            ind, dist = ind_[0], dist_[0]
            N_k2_i[i] = nc_nct_union_i[ind[0]]
            N_k2_d[i] = nc_nct_union_d[ind[0]]

        # 3:
        N_k2_dct = {index: neighbors for index, neighbors in N_k2_i.items()}
        Sc_cl = _nbdos(Sc, self.k2, N_k2_dct, self.rTh, self.nTh)

        # 4:
        OiC = Sc[Sc_cl != 0]
        TiC = Sc[Sc_cl == 0]

        # 5:
        Sw = defaultdict(dict)
        for i in TiC:
            for j in N_c_k1_i[i]:
                # 5: 1)
                if j in OiC and j in N_c_k1_i and i in N_c_k1_i[j]:
                    Sw[i][j] = 1 + self.w1 / np.e
                # 5: 2)
                elif j in N_c_k1_i and i in N_c_k1_i[j] and j in Sw and i in \
                        Sw[j]:
                    Sw[i][j] = Sw[j][i]
                # 5: 3)
                else:
                    dis_ij = self._pairwise_distance(X[i], X[j])
                    Ss = _compute_ss(Fs_i, Fs_d, i, dis_ij)
                    N_pn = _dpn(X, i, j, Ss)
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
                    Sw[i][j] = 1.0 / np.exp(exponent1) + self.w1 * np.exp(
                        exponent2)

        # 6:
        P = defaultdict(dict)
        for i in TiC:
            # 6: 1)
            for j in Sw[i]:
                dis_ij = self._pairwise_distance(X[i], X[j])
                Ss = _compute_ss(Fs_i, Fs_d, i, dis_ij)
                PN = _dpn(X, i, j, Ss)
                if (y[PN] == self.c).all():
                    break
            else:
                if i not in N_c_k1_i[i]:
                    N_c_k1_i[i] = np.append(N_c_k1_i[i], i)
                    N_c_k1_d[i] = np.append(N_c_k1_d[i], 0.)
                Sw[i][i] = 1 + self.w1 / np.e
            # 6: 2)
            for j in Sw[i]:
                P[i][j] = Sw[i][j] / sum(Sw[i][l] for l in N_c_k1_i[i])

        # 7:
        N_syn = dict()
        div, mod = divmod(self.zeta, Sc.shape[0])
        for i in Sc:
            N_syn[i] = div + (mod > 0)
            mod -= 1

        # 8:
        SI = []
        for i in Sc:
            # 8: 3)
            for _ in range(N_syn[i]):
                # 8: 1)
                if i in TiC:
                    p = [P[i][j] for j in N_c_k1_i[i]]
                    j = np.random.choice(N_c_k1_i[i], size=1, p=p)
                else:
                    j = np.random.choice(N_c_k1_i, size=1)
                # 8: 2)
                si = X[i] + (X[j] - X[i]) * np.random.rand(*X[j].shape)
                SI.append(si)
        X_resampled = np.concatenate([X, np.concatenate(SI, 0)], 0)
        y_resampled = np.concatenate([y, [self.c] * len(SI)], 0)
        if self.shuffle:
            X_resampled, y_resampled = array_util.shuffle(X_resampled,
                                                          y_resampled)
        return X_resampled, y_resampled
