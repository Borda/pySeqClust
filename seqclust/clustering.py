"""
Module with clustering functions

"""

import logging

import numpy as np

import seqclust.distances as sq_dist
import seqclust.utilities as utils


def _check_inputs(sequences, importance=None):
    assert len(sequences) == len(importance), 'length of inputs does not match,' \
                                              'sequences: %i & importance: %i' \
                                               % (len(sequences), len(importance))
    return sequences, importance


class Clustering(object):
    """ Abstract class for any clustering

    >>> clust = Clustering()
    >>> repr(clust)  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '<....Clustering object at ...>\\n Clustering(_fitted=False)'
    """

    def __init__(self):
        """ constructor """
        self._fitted = False
        self.clusters_ = []
        self.cluster_pivots_ = []
        self.labels_ = []
        self.inter_dist_ = []

    def __repr__(self):
        """ representation with printing cluster parameters """
        s = '<{0}.{1} object at {2}>'.format(self.__module__,
                                             type(self).__name__,
                                             hex(id(self)))
        obj_vars = vars(self)
        str_vars = ', '.join(['{0}={1}'.format(k, obj_vars[k])
                              for k in sorted(obj_vars) if not k.endswith('_')])
        s += '\n {0}({1})'.format(type(self).__name__, str_vars)
        return s

    def _init_fitting(self, sequences, importance):
        """ initialise internal parameters with checking come assumptions

        :param [] sequences:
        :param [] importance:
        """
        logging.info('initialize clustering...')
        if importance is None:
            importance = [sq_dist.compute_importance(s) for s in sequences]
        _check_inputs(sequences, importance)
        self.sequences = sequences
        self.importance = importance
        self.clusters_ = [[i] for i in range(len(sequences))]
        self.cluster_pivots_ = list(range(len(sequences)))
        self.labels_ = list(range(len(sequences)))
        self.inter_dist_ = [0] * len(sequences)
        self.linked_pairs_ = []

    def fit(self, sequences, importance=None):
        """ perform clustering on given sequences

        :param [] sequences: list of sequences
        :param [] importance: importance of particular sample in range (0, 1)
        """
        self._init_fitting(sequences, importance)
        logging.warning('empty abstract Fit() function for %i (%i) sequences'
                        % (len(sequences), len(importance)))
        self._fitted = True


def find_matrix_min(matrix):
    """ find matrix minimal values

    :param ndarray matrix:
    :return float, [(int, int)]:

    >>> np.random.seed(0)
    >>> mx = np.round(np.random.random((3, 4)), 3)
    >>> mx
    array([[0.549, 0.715, 0.603, 0.545],
           [0.424, 0.646, 0.438, 0.892],
           [0.964, 0.383, 0.792, 0.529]])
    >>> find_matrix_min(mx)
    (0.383, [(2, 1)])
    >>> find_matrix_min(np.zeros((2, 2)))
    (0.0, [(0, 0), (0, 1), (1, 1)])
    """
    d_min = np.min(matrix)
    dist_min = tuple(zip(*np.where(matrix <= d_min)))

    dist_min_sim = []
    for p in dist_min:
        if not p[::-1] in dist_min_sim:
            dist_min_sim.append(p)

    return d_min, dist_min_sim


class AgglomerativeClustering(Clustering):
    """ Agglomerative clustering

    >>> clust = AgglomerativeClustering(nb_clusters=2,
    ...                                 fn_affinity=sq_dist.sequence_distance)
    >>> repr(clust)  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '<....AgglomerativeClustering object at ...>\\n
    AgglomerativeClustering(_fitted=False,
        fn_affinity=<function sequence_distance at ...>,
        inter_affinity=None, nb_clusters=2)'
    >>> ss = ['Hi there, how are you?', 'hi how are you', 'hi are you there...',
    ...       'i like to sing', 'I am going to sing', 'hi where you are']
    >>> ss = [utils.sentence_tokenize(s) for s in ss]
    >>> clust.fit(ss)
    >>> repr(clust)  #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '<....AgglomerativeClustering object at ...>\\n
    AgglomerativeClustering(_fitted=True,
        fn_affinity=<function sequence_distance at ...>,
        inter_affinity=None, nb_clusters=2)'
    >>> np.array(ss)[clust.cluster_pivots_]
    array([list(['hi', 'there', 'how', 'are', 'you']),
           list(['i', 'like', 'to', 'sing'])], dtype=object)
    >>> clust.labels_
    array([0, 0, 0, 1, 1, 0])
    >>> np.round(clust.inter_dist_, 3)
    array([0.098, 0.1  ])
    """

    def __init__(self, fn_affinity, nb_clusters=5, inter_affinity=None):
        """ initialise the clustering with parameters

        :param func fn_affinity:
        :param int nb_clusters:
        :param float inter_affinity:
        """
        self.nb_clusters = nb_clusters
        self.fn_affinity = fn_affinity
        self.inter_affinity = inter_affinity
        super(AgglomerativeClustering, self).__init__()

    def _check_stop_crit(self):
        """ check stopping criterion in agglomerations

        :return bool: False if it stopes
        """
        b_nb = len(self.clusters_) > self.nb_clusters
        b_dist = max(self.inter_dist_) < self.inter_affinity \
            if self.inter_affinity is not None else True
        return b_nb and b_dist

    def fit(self, sequences, importance=None):
        """ perform clustering on given sequences

        :param [] sequences: list of sequences
        :param [] importance: importance of particular sample in range (0, 1)
        """
        self._init_fitting(sequences, importance)
        logging.info('compute precomputed distances')
        self._mx_dist = sq_dist.compute_seq_distances(self.sequences,
                                                      self.fn_affinity)
        # set inf to distances to itself
        for i in range(len(self._mx_dist)):
            self._mx_dist[i, i] = np.Inf
        mx_dist_iter = self._mx_dist.copy()

        logging.info('start agglomerating')
        while self._check_stop_crit():

            _, pairs = find_matrix_min(mx_dist_iter)

            # merge clusters
            self._merge_clusters(pairs)

            self._update_cluster_pivots()
            pivs = np.asarray(self.cluster_pivots_)
            mx_dist_iter = self._mx_dist[pivs, :][:, pivs]
            self._update_labels_inter_dist()

        logging.info('finish cleaning')
        self._fitted = True
        del self._mx_dist
        del self.sequences
        del self.importance

    def _merge_clusters(self, pairs):
        """ merge clusters if given

        :param [(int, int)] pairs: cluster pairs for merging
        """
        for c1, c2 in pairs:
            if not self._check_stop_crit():
                break
            self.clusters_[c1] += self.clusters_[c2]
            self.clusters_[c2] = []
            p1 = self.cluster_pivots_[c1]
            p2 = self.cluster_pivots_[c2]
            self.linked_pairs_.append((p1, p2))
        self.clusters_ = [c for c in self.clusters_ if len(c) > 0]

    def _update_labels_inter_dist(self):
        """ update labels for sequences and interiar cluster distance
        """
        self.inter_dist_ = [0] * len(self.clusters_)
        for idx, clust in enumerate(self.clusters_):
            for i in clust:
                self.labels_[i] = idx
            if len(clust) > 1:
                _, idist = self._compute_pivot_inter_dist(clust)
                self.inter_dist_[idx] = idist / len(clust)
        self.labels_ = np.array(self.labels_)
        self.inter_dist_ = np.array(self.inter_dist_)

    def _compute_pivot_inter_dist(self, clust):
        """ compute pivot (most representative sample) and inter distance
        for each particular cluster

        :param [int] clust: index of samples in cluster
        """
        assert hasattr(self, '_mx_dist'), 'missing precomputed distances'
        clust = np.asarray(clust)
        mx_inter = self._mx_dist[clust, :][:, clust]
        # remove infs
        mx_inter[np.isinf(mx_inter)] = 0
        # mean inter cluster distance
        idist = np.mean(mx_inter)
        inter = np.sum(mx_inter, axis=0)
        # samples with low importance will increase distance
        inter /= np.asarray(self.importance)[clust]
        # take the sample with smallest sum distance
        piv = clust[np.argmin(inter)]
        return piv, idist

    def _update_cluster_pivots(self):
        """
        compute pivots for each cluster as sample with min dist to others
        """
        self.cluster_pivots_ = []
        for clust in self.clusters_:
            if len(clust) == 1:
                piv = clust[0]
            else:
                piv, _ = self._compute_pivot_inter_dist(clust)
            self.cluster_pivots_.append(piv)
        self.cluster_pivots_ = np.array(self.cluster_pivots_)
