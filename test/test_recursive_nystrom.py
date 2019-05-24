import numpy as np
import scipy.linalg as sp_linalg
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
        X = np.concatenate([np.random.multivariate_normal(mean=[50, 10], cov=np.eye(2), size=(n1,)),
                            np.random.multivariate_normal(mean=[-70, -70], cov=np.eye(2), size=(n2,)),
                            np.random.multivariate_normal(mean=[90, -40], cov=np.eye(2), size=(n3,))], axis=0)
        y = np.concatenate([np.ones((n1,))*1,
                            np.ones((n2,))*2,
                            np.ones((n3,))*3])
        self.X = X
        self.y = y

    def test_spectral_norm(self):
        n_components = 10
        K = gauss(self.X, row_idx=np.arange(self.X.shape[0]), col_idx=np.arange(self.X.shape[0]))
        w, v = sp_linalg.eigh(K, eigvals=(K.shape[0]-n_components, K.shape[0]-1))

        K_optim_approx = np.dot(v,np.dot(np.diag(w), v.T))
        indices = recursiveNystrom(self.X, n_components=n_components)
        K_basis = gauss(self.X, row_idx=indices, col_idx=indices, gamma=0.01)
        K_approx = gauss(self.X, row_idx=np.arange(self.X.shape[0]), col_idx=indices, gamma=0.01)

        S, U = np.linalg.eigh(K_basis)  # for PSD matrices eigendecomposition == SV-decomposition
        S = np.maximum(S, 1e-12)  # regularisation
        normalization = np.dot(U / np.sqrt(S), U.T)
        feature_map = np.dot(K_approx, normalization)
        K_approx = np.dot(feature_map,feature_map.T)

        norm_approx = np.linalg.norm(K - K_approx, 'fro')
        norm_optim_approx = np.linalg.norm(K - K_optim_approx, 'fro')
        print("Norm approx: %.2f" % norm_approx)
        print("Norm optim: %.2f" % norm_optim_approx)
if __name__ == "__main__":
    unittest.main()
