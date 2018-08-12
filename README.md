# Sequence Clustering

[![Build Status](https://travis-ci.org/Borda/pySeqClust.svg?branch=master)](https://travis-ci.org/Borda/pySeqClust)
[![CircleCI](https://circleci.com/gh/Borda/pySeqClust/tree/master.svg?style=svg)](https://circleci.com/gh/Borda/pySeqClust/tree/master)
[![Build status](https://ci.appveyor.com/api/projects/status/3v0q2514jabbap7f?svg=true)](https://ci.appveyor.com/project/Borda/pyseqclust)
[![codecov](https://codecov.io/gh/Borda/pySeqClust/branch/master/graph/badge.svg)](https://codecov.io/gh/Borda/pySeqClust)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/12f029b7342c478c982854565ff7c5f4)](https://www.codacy.com/project/Borda/pySClust/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Borda/pySClust&amp;utm_campaign=Badge_Grade_Dashboard)

Clustering on consecutive sequences like sentences where the order matters.
The core is based on [Dynamic Time Warping](https://en.wikipedia.org/wiki/Dynamic_time_warping) for comparing difference between individual sequences and [agglomerative clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) which construct the clusters from bottom-up.

Let's have sample set of sequences and [perform clustering](notebooks/sample_text.ipynb).
```
Hi there, how are you?
hi how you are
i like to sing
I am going to sing
hi where you are
hi are you there...
do you sing???
```
with binary distance between block and sett 3 clusters we bot following results:

|   sentence    |	clusters	|   internal dist.  |
|---|:---:|:---:|
|   hi how you are  |	[0, 1, 4, 5]    |	0.095313    |
|   i like to sing  |	[2, 3]  |	0.100000    |
|   do you sing |	[6] |	0.000000    |


The agglomerative clustering has two stop criteria, one is number of clusters and second is maximal internal distance inside cluster.
The "pivot" the most representative sample from cluster is selected as such with minimal distance to all others inside own cluster.
