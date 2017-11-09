import scipy
import numpy as np
from copy import deepcopy


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
    return U[:, sum(abs(s) > 0):]


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
