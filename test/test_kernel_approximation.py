import numpy as np
import scipy.linalg as sp_linalg
import unittest
from feature_maps import FeatureMap
from kernel_functions import KernelFunction


def gaussian_mixture(n_mixtures=3, d=5, random_state = None):
    space_lower_bound = -100*np.ones((d))
    space_upper_bound = 100*np.ones((d))

    rng = np.random.RandomState(random_state)
    mean_list = list()
    cov_list = list()
    for i in range(n_mixtures):
        mean_list.append(rng.uniform(low=space_lower_bound,
                             high=space_upper_bound,))
        c = rng.gamma(shape=2, scale=2, size=(d,d))
        cov_list.append(c + c.T)

    while True:
        i = rng.choice(n_mixtures)
        yield rng.multivariate_normal(mean=mean_list[i], cov=cov_list[i])


class test_kernel_approximation(unittest.TestCase):

    def setUp(self):

        self.X = np.stack([sample for _ , sample in zip(range(2000), gaussian_mixture())])

    def test_feature_map(self):
        n_components = 21
        kernel_type = "rbf"
        kernel_params = {"gamma": .059}

        rls_nystrom_fm = FeatureMap.construct(kernel=kernel_type,
                                              kernel_params=kernel_params,
                                              algorithm="rls_nystroem",
                                              n_components=n_components)
        uniform_nystrom_fm = FeatureMap.construct(kernel=kernel_type,
                                                  kernel_params=kernel_params,
                                                  algorithm="uniform_nystroem",
                                                  n_components=n_components)

        kernel_func = KernelFunction.construct(kernel_type, **kernel_params)

        rls_feature_map = rls_nystrom_fm.fit_transform(self.X)
        rls_gram = np.dot(rls_feature_map, rls_feature_map.T)

        uniform_feature_map = uniform_nystrom_fm.fit_transform(self.X)
        uniform_gram = np.dot(uniform_feature_map, uniform_feature_map.T)

        gram = kernel_func(self.X)
        W, V = np.linalg.eigh(gram)
        V = V[:,-n_components:]
        W = W[-n_components:]
        low_rank_gram = np.dot(V, np.dot(np.diag(W), V.T))

        single_cluster = np.linalg.norm(gram, 2)
        #three_clusters = np.linalg.norm(low_rank_gram, 'fro')
        #noise = np.linalg.norm(gram-low_rank_gram, 'fro')

        norm_uniform_approx = np.linalg.norm(gram - uniform_gram, 'fro')
        norm_rls_approx = np.linalg.norm(gram - rls_gram, 'fro')
        norm_optim_approx = np.linalg.norm(gram - low_rank_gram, 'fro')

        #print("Cluster signal balance: %.2f" % (single_cluster*np.sqrt(3)/three_clusters))
        #print("Signal to Noise: %.2f" % (three_clusters**2/noise**2))
        print("Norm uniform approx: %.4f" % (norm_uniform_approx**2/3/single_cluster**2))
        print("Norm RLS approx: %.4f" % (norm_rls_approx**2/3/single_cluster**2))
        print("Norm optim: %.4f" % (norm_optim_approx**2/3/single_cluster**2))
if __name__ == "__main__":
    unittest.main()
