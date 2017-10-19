import numpy as np
from scipy.stats import special_ortho_group
from sklearn.utils import check_array


def get_gaussian_inputs(sample_size, eig_values, eig_vectors=None, mean=None):
    # generate covariance mat
    dim = len(eig_values)
    eig_values = np.diag(eig_values)
    if eig_vectors is None:
        eig_vectors = special_ortho_group.rvs(dim)
    else:
        eig_vectors = check_array(eig_vectors)
        if eig_vectors.shape != (dim, dim):
            raise ValueError('eig_vectors shape must be ({0},{0})'.format(dim))
    cov = np.dot(np.dot(eig_vectors, eig_values), eig_vectors.T)
    # generate centered inputs
    if mean is None:
        mean = np.zeros(dim)
    X = np.random.multivariate_normal(mean, cov, sample_size)
    return X


def get_tanh_targets(X, coefs, bias=0, noise_std=0.05):
    if X.shape[1] != len(coefs):
        raise ValueError(
            'Dimensionality of input ({}) and coefs ({}) are mismatched')
    y = np.tanh(np.dot(X, coefs) + bias)
    y += + noise_std * np.random.randn(X.shape[0])
    return y
