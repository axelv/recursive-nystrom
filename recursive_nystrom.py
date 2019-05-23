# axelv 16/05/2019
import numpy as np
import scipy.linalg as spl
import time
from tqdm import tqdm
import gc


def gauss(X: np.ndarray, row_idx=None, Y: np.ndarray=None, col_idx=None, gamma=0.05):
    # todo make this implementation more python like!
    if Y is None:
        Y = X
    if row_idx is None and col_idx is None:
        row_idx = np.arange(X.shape[0])
        col_idx = np.arange(Y.shape[0])

    assert len(row_idx.shape) < 2
    assert col_idx is None or len(col_idx.shape) < 2

    if col_idx is None or col_idx.size == 0:
        Ksub = np.ones((row_idx.size, 1))
    else:
        nsq_rows = np.sum(X[row_idx, :] ** 2, axis=1, keepdims=True)
        nsq_cols = np.sum(Y[col_idx, :] ** 2, axis=1, keepdims=True)
        Ksub = nsq_rows - np.dot(X[row_idx, :], Y[col_idx, :].T * 2)
        Ksub = nsq_cols.T + Ksub
        Ksub = np.exp(-gamma * Ksub)

    return Ksub


def uniformNystrom(X, n_components: int, kernel_func=gauss):
    indices = np.random.choice(X.shape[0], n_components)
    C = kernel_func(X, row_idx=np.arange(X.shape[0]), col_idx=indices)
    SKS = C[indices, :]
    W = np.linalg.inv(SKS + 10 - 6 * np.eye(n_components))

    return C, W


def recursiveNystrom(X, n_components: int, kernel_func=gauss, accelearted_flag=False, random_state=None, other=None):
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
    k_diag = kernel_func(X, row_idx=np.arange(X.shape[0]),
                            col_idx=None)

    # Main recursion, unrolled for efficiency
    # todo: replace with reversed(enumeration(size_list))
    for l in reversed(range(n_levels)):

        # indices of current uniform sample
        current_indices = perm[0:size_list[l]]
        # build sampled kernel

        # all rows and sampled columns
        KS = kernel_func(X, row_idx=current_indices,
                            col_idx=indices)
        SKS = KS[sample, :] # sampled rows and sampled columns

        # optimal lambda for taking O(k log(k)) samples
        if k >= SKS.shape[1]:
            # for the rare chance we take less than k samples in a round
            lmbda = 10e-6
            # don't set to exactly 0 to avoid stability issues
        else:

            lmbda = (np.sum(np.diag(SKS) * weights ** 2) \
                    - np.sum(spl.eigvalsh(SKS * weights * weights.T, eigvals=(SKS.shape[0]-k, SKS.shape[0]-1))))/k

            #lmbda = spl.eigvalsh(SKS * weights * weights.T, eigvals=(0, SKS.shape[0]-k-1)).sum()/k
            # calculate the n-k smallest eigenvalues

        # compute and sample by lambda ridge leverage scores
        if l != 0:
            R = np.linalg.inv(SKS + np.diag(lmbda * weights ** (-2)))
            # max(0, . ) helps avoid numerical issues, unnecessary in theory
            leverage_score = np.minimum(1.0, n_oversample * (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - np.sum(np.dot(KS, R) * KS, axis=1))))
            # on intermediate levels, we independently sample each column
            # by its leverage score. the sample size is n_components in expectation
            sample = np.where(rng.uniform(size=size_list[l]) < leverage_score)[0]
            # with very low probability, we could accidentally sample no
            # columns. In this case, just take a fixed size uniform sample
            if sample.size == 0:
                leverage_score[:] = n_components / size_list[l]
                sample = rng.choice(size_list[l], size=n_components, replace=False)
            weights = np.sqrt(1. / leverage_score[sample])

        else:
            R = np.linalg.inv(SKS + np.diag(lmbda * weights ** (-2)))
            leverage_score = np.minimum(1.0, (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - np.sum(np.dot(KS, R) * KS, axis=1))))
            p = np.maximum(leverage_score, 1e-10) # avr: make sure that there are enough non-zero entries to choose n_components
            p = p/p.sum()

            sample = rng.choice(X.shape[0], size=n_components, replace=False, p=p)
        indices = perm[sample]

    return indices


# Below the copyright info that came with the original MATLAB implementation
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


# Small check to test if the algorithms output makes sense
if __name__ == "__main__":
    n1 = 100
    n2 = 5000
    n3 = 4900
    n = np.asarray([n1, n2, n3])
    X = np.concatenate([np.random.multivariate_normal(mean=[50, 10], cov=np.eye(2), size=(n1,)),
                        np.random.multivariate_normal(mean=[-70, -70], cov=np.eye(2), size=(n2,)),
                        np.random.multivariate_normal(mean=[90, -40], cov=np.eye(2), size=(n3,))], axis=0)
    y = np.concatenate([np.ones((n1,)) * 1,
                        np.ones((n2,)) * 2,
                        np.ones((n3,)) * 3])

    y_list = list()

    gauss(X, )
    for i in range(1000):
        indices = recursiveNystrom(X, n_components=10, kernel_func=lambda *args, **kwargs: gauss(*args, **kwargs, gamma=0.001))
        y_list.append(y[indices])

    y_total = np.concatenate(y_list)
    u,c = np.unique(y_total, return_counts=True)
    print("Real balance:", n/n.sum())
    print("RLS balance:", c/c.sum())
