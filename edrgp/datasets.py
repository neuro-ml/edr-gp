import numpy as np
from scipy.stats import special_ortho_group
from sklearn.utils import check_array
from math import pi


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
    y += noise_std * np.random.randn(X.shape[0])
    return y


def get_beta_inputs(sample_size, ndim, tau=1):
    "Generates data from 2*Beta(1, tau)-1"
    return 2*np.random.beta(1, tau, size=(sample_size, ndim))-1


def get_edr_target(X, sigma=None):
    """"Generates target for different dimensions

    n=1 => g(u) = u sin(sqrt(5) u)
    n=2 => g(u1, u2) = (u1^3+u2)(u1-u2^3)
    n=3 => g(u1, u2, u3) = (u1^3+u2)(u1-u2^3)+u3*u3_coef
    """
    effective_ndim = X.shape[1]

    if effective_ndim == 1:
        result = np.array([u*np.sin(np.sqrt(5)*u) for u in X])
    if effective_ndim == 2:
        result = np.array([(u1 ** 3+u2)*(u1-u2 ** 3) for u1, u2 in X])
    if effective_ndim == 3:
        result = np.array([(u1 ** 3+u2)*(u1-u2 ** 3)+u3 for u1, u2, u3 in X])
    result = result.ravel()
    if sigma is not None:
        result += sigma * np.random.randn(result.size)
    return result


def get_branin_targets(X, noise_std=None):
    '''Branin function generator
    It's a smooth 2D function which is frequently used as a target for
    testing of optimization algorithms.
    For details see https://www.sfu.ca/~ssurjano/branin.html

    Parameters
    ----------
    X : np.array
        The input data. There are two requirements:
         - must be a 2D array with a shape [n_samples, 2]
         - both columns should be in range [0, 1]
    noise_std : float or None
        Standard deviation of iid Gaussian noise which will be added to outputs

    Returns
    -------
    y : np.array
        Vector of target, [n_samples, ]
    '''
    # ToDo: check X

    # params
    a, b, c = 1, 5.1 / (4 * pi**2), 5 / pi
    r, s, t = 6, 10, 1 / (8 * pi)
    # x0 ∈ [-5, 10], x1 ∈ [0, 15].
    x0 = 15 * X[:, 0] - 5
    x1 = 15 * X[:, 1]
    y = a * (x1 - b * x0**2 + c * x0 - r)**2 + s * (1 - t) * np.cos(x0) + s
    if noise_std is not None:
        y += noise_std * np.random.randn(X.shape[0])
    return y
