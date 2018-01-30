import scipy
import numpy as np
from copy import deepcopy
from sklearn.utils import check_array
from sklearn.base import TransformerMixin


def ort_space(A):
    """Retutn orthogonal space for matrix a

    Parametrs
    ---------
    A: np.ndarray
        The matrix [n_features, n_components] for which we want to find
        orthogonal space

    Returns
    -------
    U: np.ndarray
        The matrix [n_features, n_features - n_components] which columns are
        basis in orthogonal space
    """
    U, s, _ = scipy.linalg.svd(a=A, full_matrices=True)
    return U[:, sum(abs(s) > 1e-10):]


def prepare_data(X, V):
    X_copy = deepcopy(X)
    X_copy -= np.mean(X_copy, axis=0)

    if len(V.shape) == 1:
        V_copy = np.array(V, ndmin=2).T
        dim = 1
    else:
        V_copy = deepcopy(V)
        dim = V.shape[1]

    return X_copy, V_copy, dim


def _subspace_variance(X, V):
    F = ort_space(V)
    tot_var = np.trace(np.dot(X.T, X))

    D = np.dot(F, F.T)
    var = np.trace(np.dot(X, np.dot(D, X.T)))

    return (tot_var - var) / (X.shape[0] - 1), 1 - var / tot_var


def subspace_variance(X, V):
    X_copy, V_copy, dim = prepare_data(X, V)
    indexes = []
    expl_var = []
    expl_var_ratio = []

    for j in range(dim):
        var_ratio = []
        var = []

        for i in range(dim):

            if i not in indexes:
                _var = _subspace_variance(X_copy, V_copy[:, indexes + [i]])
                var.append(_var[0])
                var_ratio.append(_var[1])
            else:
                var.append(-1)
                var_ratio.append(-1)

        idx = np.argsort(var_ratio)[::-1][0]
        indexes.append(idx)
        expl_var.append(var[idx])
        expl_var_ratio.append(var_ratio[idx])

    return V_copy[:, indexes], np.array(expl_var), np.array(expl_var_ratio)


class CustomPCA(TransformerMixin):
    """PCA without centering and scaling

    Linear dimensionality reduction using Singular Value 
    Decomposition of the data to project it to a lower dimensional 
    space.

    Parameters
    ----------
    n_components : int, float, None or string
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        if ``0 < n_components < 1``, select the number
        of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.

    subspace_variance_ : array, shape (n_components,)
        The amount of variance contained in each of the selected components.

        Equal to n_components largest eigenvalues
        of the covariance matrix of X.

    subspace_variance_ratio_ : array, shape (n_components,)
        Percentage of variance contained in each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of subspace variances is equal to 1.0.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored.

        Returns
        -------
        self : object
        """
        X_copy = check_array(X, copy=True)

        U, S, Vh = np.linalg.svd(X_copy)
        subspace_var_ratio = S**2 / np.sum(S**2)

        if self.n_components is None:
            n_components = X_copy.shape[1]
        if (isinstance(self.n_components, int)
                and 0 < self.n_components <= X_copy.shape[1]):
            n_components = self.n_components
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            n_components = np.sum(np.cumsum(expl_var_ratio) <
                                  self.n_components, dtype=int) + 1
        n_components = min(X_copy.shape[0], n_components)

        self.components_ = Vh[:n_components, :]
        self.subspace_variance_ = (S**2)[:n_components]
        self.subspace_variance_ratio_ = subspace_var_ratio[:n_components]

        return self

    def transform(self, X):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously 
        extracted from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        return X.dot(self.components_.T)
