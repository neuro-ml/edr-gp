import scipy
import numpy as np
from copy import deepcopy
from sklearn.utils import check_array
from sklearn.base import TransformerMixin, BaseEstimator


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


def subspace_variance_ratio(X, V):
    """Compute subspace variance for V in X

    If V is not orthonormalized then it will return only one 
    value: subspace variance for the whole matrix V.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)

    V : array, shape (n_feaures, n_components)
        Projector components

    Returns
    -------
    subspace_variance_ : array, shape (n_components, )
        Subspace variance for each column of V (or for the whole V 
        if it is not orthonormalized)
    subspace_variance_ratio_ : array, shape (n_components, )
        Subspace variance ratio for each column of V
    """
    if np.allclose(np.dot(V.T, V), np.eye(V.shape[1])):
        subspace_variance_ = np.linalg.norm(X.dot(V), axis=0)
    else:
        V_orthonormalized = np.linalg.qr(V)[0]
        subspace_variance_ = np.linalg.norm(X.dot(V_orthonormalized))

    subspace_variance_ratio_ = (subspace_variance_/np.linalg.norm(X)) ** 2
    return subspace_variance_, subspace_variance_ratio_


def discrepancy(B, V):
    """Measure of difference between two subspaces

    The discrepancy is measuared by the following formula:
    ||BB^T(I - VV^T)||_F/d, where d denotes number of true components and
    I is an identity matrix with shape n_features.

    Parameters
    ----------
    B : array, shape (n_features, n_components_true)
        The true projector on subspace

    V : array, shape (n_features, n_components)
        The estimated projector on subspace

    Returns
    -------
    """
    return np.linalg.norm(np.dot(B.dot(B.T),
                                 (np.eye(B.shape[0]) - 
                                                    V.dot(V.T)))) / B.shape[1]


class CustomPCA(BaseEstimator, TransformerMixin):
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
