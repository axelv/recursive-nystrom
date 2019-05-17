# axelv 16/05/2019
import numpy as np
import scipy.linalg as spl
import time
from tqdm import tqdm
import gc

def gauss(X: np.ndarray, row_idx, col_idx, gamma=1):
    assert len(row_idx.shape) < 2
    assert col_idx is None or len(col_idx.shape) < 2

    if col_idx is None or col_idx.size == 0:
        Ksub = np.ones((row_idx.size, 1))
    else:
        nsq_rows = np.sum(X[row_idx, :] ** 2, axis=1, keepdims=True)
        nsq_cols = np.sum(X[col_idx, :] ** 2, axis=1, keepdims=True)
        Ksub = nsq_rows - np.dot(X[row_idx, :], X[col_idx, :].T * 2)
        Ksub = nsq_cols.T + Ksub
        Ksub = np.exp(-gamma * Ksub)

    return Ksub


def uniformNystrom(X, n_components: int, kernel_func=gauss):
    sample = np.random.choice(X.shape[0], n_components)
    C = kernel_func(X, np.arange(X.shape[0]), sample)
    SKS = C[sample, :]
    W = np.linalg.inv(SKS + 10 - 6 * np.eye(n_components))

    return C, W


def recursiveNystrom(X, n_components: int, kernel_func=gauss, accelearted_flag=False, random_state=None):
    '''

    :param X:
    :param n_components:
    :param kernel_func:
    :param accelearted_flag:
    :param random_state:
    :return:
    '''

    rng = np.random.RandomState(random_state)

    n_oversample = np.log(n_components)
    k = np.ceil(n_components / (4 * n_oversample)).astype(np.int)
    n_levels = np.ceil(np.log(X.shape[0] / n_components) / np.log(2)).astype(np.int)
    perm = np.random.permutation(X.shape[0])

    # set up sizes for recursive levels
    size_list = [X.shape[0]]
    for l in range(1, n_levels):
        size_list += [np.ceil(size_list[l - 1] / 2).astype(np.int)]

    # indices of poitns selected at previous level of recursion
    # at the base level it's just a uniform sample of ~ n_component points
    sample = np.arange(size_list[-1])
    indices = perm[sample]
    weights = np.ones((indices.shape[0], 1))

    # we need the diagonal of the whole kernel matrix, so compute upfront
    k_diag = kernel_func(X, np.arange(X.shape[0]), None)

    # Main recursion, unrolled for efficiency
    # todo: replace with reversed(enumeration(size_list))
    for l in reversed(range(n_levels)):

        # indices of current uniform sample
        current_indices = perm[0:size_list[l]]
        # build sampled kernel
        KS = kernel_func(X, current_indices, indices)
        SKS = KS[sample, :]

        # optimal lambda for taking O(k log(k)) samples
        if k >= SKS.shape[1]:
            # for the rare chance we take less than k samples in a round
            lmbda = 10e-6
            # don't set to exactly 0 to avoid stability issues
        else:

            lmbda = np.sum(np.diag(SKS) * weights ** 2) \
                    - np.sum(np.abs(spl.eigvalsh(SKS * weights.T,
                                                 eigvals=(k, SKS.shape[0]-1))))/k

        # compute and sample by lambda ridge leverage scores
        R = np.linalg.inv(SKS + np.diag(lmbda * weights ** (-2)))
        # max(0, . ) helps avoid numerical issues, unnecessarry in theory
        leverage_score = np.minimum(1.0, n_oversample * (1 / lmbda) * np.maximum(0.0, (
                k_diag[current_indices, 0] - np.sum(np.dot(KS, R) * KS, axis=1))))
        if l != 0:
            # on intermediate levels, we independently sample each column
            # by its leverage score. the sample size is n_components in expectation
            # todo: check if this can be replaced with np.choice
            sample = np.where(rng.uniform(size=size_list[l]) < leverage_score)[0]
            # with very low probability, we could accidentally sample no
            # columns. In this case, just take a fixed size uniform sample
            if sample.size == 0:
                leverage_score[:] = n_components / size_list[l]
                sample = rng.choice(size_list[l], size=n_components, replace=False)
            weights = np.sqrt(1. / leverage_score[sample])

        else:
            sample = rng.choice(X.shape[0], size=n_components, replace=False, p=leverage_score/leverage_score.sum())
        indices = perm[sample]

    # build final Nyrstrom approximation
    # pinv or inversion with slight regularization helps stability
    C = kernel_func(X, np.arange(X.shape[0]), indices)
    SKS = C[indices, :]
    W = np.linalg.inv(SKS + np.eye(n_components) * 10e-6)

    return C, W


# -------------------------------------------------------------------------------------
# Copyright (c) 2017 Christopher Musco, Cameron Musco
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


if __name__ == "__main__":
    import pandas as pd

    X = pd.read_csv("covtype.csv", header=None, nrows=50000).values
    delta_t = -time.time()
    C_uni, W_uni = uniformNystrom(X, 100)
    delta_t += time.time()
    print("Total time uniform Nystrom: %s " % delta_t)

    del C_uni
    del W_uni
    gc.collect()

    delta_t = -time.time()
    C_rls, W_rls = recursiveNystrom(X, 100)
    delta_t += time.time()
    print("Total time uniform Nystrom: %s " % delta_t)
