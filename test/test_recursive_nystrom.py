import numpy as np
import scipy.linalg as sp_linalg
import matplotlib.pyplot as plt
import unittest
from recursive_nystrom import gauss, recursiveNystrom


def gaussian_mixture(n_mixtures=3, d=2, prior=None, random_state = None):

    assert prior.size == n_mixtures

    space_lower_bound = -100*np.ones((d))
    space_upper_bound = 100*np.ones((d))

    rng = np.random.RandomState(random_state)
    mean = list()
    cov = list()
    for i in range(n_mixtures):
        mean.append(rng.uniform(low=space_lower_bound,
                             high=space_upper_bound,))
        c = rng.gamma(shape=2, scale=2, size=(d,d))
        cov.append(c + c.T)

    while True:
        i = rng.choice(n_mixtures, p=prior)
        yield np.concatenate((rng.multivariate_normal(mean=mean[i], cov=cov[i]), np.asarray([i])),)


class testRecursiveNystrom(unittest.TestCase):

    def setUp(self):

        n1 = 100
        n2 = 5000
        n3 = 4900
        X = np.concatenate([np.random.multivariate_normal(mean=[50, 10], cov=np.eye(2)*0.1, size=(n1,)),
                            np.random.multivariate_normal(mean=[-70, -70], cov=np.eye(2)*0.1, size=(n2,)),
                            np.random.multivariate_normal(mean=[90, -40], cov=np.eye(2)*0.1, size=(n3,))],
                           axis=0)
        y = np.concatenate([np.ones((n1,))*1,
                            np.ones((n2,))*2,
                            np.ones((n3,))*3])
        self.X = X
        self.y = y


    def test_leverage_scores(self):
        N = 10000
        ls_array = np.zeros((N, self.X.shape[0]))

        for i in range(N):
            indices, ls = recursiveNystrom(self.X, n_components=12, lmbda_0=1e-6, y=self.y, random_state=1111)
            if ls.sum()/2.5 > 4:
                print("Iteration %d has an effective dimension of %.2f ." % ls.sum())
            count, idx = np.unique(self.y[indices], return_counts=True)
            if np.unique(self.y[indices], return_counts=True)[1].shape[0] < 3:
                print("Iteration has only samples for clusters %s, it's effective dimension is %.2f."
                      % (str(idx), ls.sum()))
            ls_array[i,:] = ls
        d_eff = ls_array.sum(axis=1)
        print("%d samples with an effective dimension that is way to high!" % np.sum(d_eff > 10))

    def test_spectral_norm(self):
        n_components = 12
        gamma = 0.01

        K = gauss(self.X, self.X, gamma=0.01)
        w, v = sp_linalg.eigh(K, eigvals=(K.shape[0]-n_components, K.shape[0]-1))
        K_lr = np.dot(v,np.dot(np.diag(w), v.T))

        # RLS
        indices = recursiveNystrom(self.X, n_components=n_components, lmbda_0=1e-6, random_state=1111)
        K_basis = gauss(self.X[indices], self.X[indices], gamma=gamma)
        K_approx = gauss(self.X, self.X[indices], gamma=gamma)

        K_rls = self.nystrom(K_basis, K_approx)

        # UNIFORM
        idx = np.random.RandomState(1111).choice(self.X.shape[0], size=n_components)
        K_basis = gauss(self.X[idx], self.X[idx], gamma=gamma)
        K_approx = gauss(self.X, self.X[idx], gamma=gamma)
        K_uniform = self.nystrom(K_basis, K_approx)

        norm_rls = np.linalg.norm(K - K_rls, 2)
        norm_lr = np.linalg.norm(K - K_lr, 2)
        norm_uniform = np.linalg.norm(K - K_uniform, 2)

        print("|| K - K_uniform || =  %.4f" % norm_uniform)
        print("|| K - K_RLS || =  %.4f" % norm_rls)
        print("|| K - K_LR || = %.4f" % norm_lr)


    @staticmethod
    def nystrom(K_basis, K_approx):

        S, U = np.linalg.eigh(K_basis)  # for PSD matrices eigendecomposition == SV-decomposition
        S = np.maximum(S, 1e-12)  # regularisation
        normalization = np.dot(U / np.sqrt(S), U.T)
        feature_map = np.dot(K_approx, normalization)
        return np.dot(feature_map,feature_map.T)
if __name__ == "__main__":
    unittest.main()
