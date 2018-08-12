"""
Module for computing distances

"""

import multiprocessing as mproc
from functools import partial

import numpy as np
import fastdtw

import seqclust.utilities as utils


NB_THREADS = max(1, int(mproc.cpu_count() * 0.8))


def elements_distance_binary(b1, b2, ignored=None):
    """ affinity measure between two blocks

    :param b1:
    :param b2:
    :return int:

    >>> elements_distance_binary('a', 'a')
    0
    >>> elements_distance_binary('a', 'b')
    1
    >>> elements_distance_binary('a', None)
    0
    """
    if b1 == ignored or b2 == ignored:
        return 0
    return int(b1 != b2)


def sequence_distance(seq1, seq2, ignored=[None], block_dist=elements_distance_binary):
    """ affinity between two layouts

    :param [] l1: sequence layout
    :param [] l2: sequence layout
    :return float:

    >>> seq1 = ['a', 'b', 'a', 'c']
    >>> sequence_distance(seq1, seq1)
    0.0
    >>> seq2 = ['a', 'a', 'b', 'a']
    >>> sequence_distance(seq1, seq2)
    0.25
    >>> l3 = ['c', None, 'd']
    >>> sequence_distance(seq1, l3)
    0.5
    """
    # remove None which crash uniques
    seqs = [b for b in seq1 + seq2 if b is not None]
    blocks = [b for b in np.unique(seqs) if b not in ignored]
    # discrete the lists
    seq1_d = [blocks.index(b) if b not in ignored else -1 for b in seq1]
    seq2_d = [blocks.index(b) if b not in ignored else -1 for b in seq2]

    # TODO: add also reverse time dynamic t -> (t-1)
    # compute the special distance measure
    _wrap_dist = partial(block_dist, ignored=-1)
    dist, match = fastdtw.dtw(seq1_d, seq2_d, _wrap_dist)
    dist = dist / float(max(len(seq1), len(seq2)))

    return dist


def wrap_distance(idx_lt, similar_distance):
    """ wrap distance computation so it can be called in parallel mode

    :param ((int, int) (obj, obj)) idx_lt:
    :param func similar_distance:
    :return (int, int), int:
    """
    idx, seqs = idx_lt
    d = similar_distance(seqs[0], seqs[1])
    return idx, d


def compute_seq_distances(sequences, affinity=sequence_distance):
    """ compute matrix of all distances

    :param [] layouts:
    :return ndarray:

    >>> ss = [['a', 'b', 'a', 'c'], ['a', 'a', 'b', 'a'], ['b', None, 'b', 'a']]
    >>> compute_seq_distances(ss)
    array([[0.  , 0.25, 0.5 ],
           [0.25, 0.  , 0.25],
           [0.5 , 0.25, 0.  ]])
    >>> ss = [['hi', 'there', 'how', 'are', 'you'],
    ...       ['hi', 'how', 'are', 'you'],
    ...       ['hi', 'are', 'you', 'there']]
    >>> compute_seq_distances(ss)
    array([[0. , 0.2, 0.6],
           [0.2, 0. , 0.5],
           [0.6, 0.5, 0. ]])
    """
    idxs = [(i, j) for i in range(len(sequences)) for j in range(i, len(sequences))]
    idx_lt = (((i, j), (sequences[i], sequences[j])) for i, j in idxs)
    dists = np.zeros((len(sequences), len(sequences)))

    _wrap_dist = partial(wrap_distance, similar_distance=affinity)
    pool = utils.NDPool(NB_THREADS)
    for idx, d in pool.imap_unordered(_wrap_dist, idx_lt):
        dists[idx[0], idx[1]] = d
        dists[idx[1], idx[0]] = d
    pool.close()
    pool.join()

    return dists
