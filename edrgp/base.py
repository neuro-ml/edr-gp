"""Base class for Effective Dimensionality Reduction"""

import numpy as np
from copy import deepcopy
from sklearn.base import TransformerMixin, clone
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from .utils import subspace_variance


class BaseEDR(TransformerMixin):
    """Base class for effective dimensionality reduction

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``predict_gradient``
        and ``fit`` methods.
    dr_transformer : objec
        A linear dimensionnality reduction method that provides
        information about new axes through ``components_`` attribute.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        New axes in feature space, representing the directions of
        maximum variance of the target.
        n_components = ``dr_transformer.n_components``
    feature_importances_ : ndarray, shape (n_components, n_features)
        The contribution of each feature to new axes.
        n_components = ``dr_transformer.n_components``
    estimator_ : object
        Estimator fitted to preprocessed data.
    dr_transformer_ : object
        Dr_transformer fitted to gradients.
    subspace_var_: array, shape (n_components, )
        Subspace variance calculated as tr(X.T * X) - tr(Y_i.T * Y_i)
        where Y_i=XU_i, U_i - orthogonal complement for components_.T[:, :i],
        i =  1, ..., n_components
        n_components = ``dr_transformer.n_components``
    subspace_var_ratio_: array, (n_components, )
        Subspace variance ratio calculated as subspace_var_/tr(X.T * X)
        n_components = ``dr_transformer.n_components``
    """

    def __init__(self, estimator, dr_transformer):
        self.estimator = estimator
        self.dr_transformer = dr_transformer

    def _check_transformer(self, transformer):
        """Check that transformer has attribute ``components_``.

        Parameters
        ----------
        transformer : object
        """
        if not hasattr(transformer, 'components_'):
            raise AttributeError('The transformer does not expose '
                                 '"components_" attribute')

    def fit(self, X, y=None, **opt_kws):
        """Fit the model with X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        self._fit_estimator(X, y, **opt_kws)
        self._fit_dr_transformer(X)
        return self

    def _fit_estimator(self, X, y, **opt_kws):
        """Fit the estimator with X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        if y is not None:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **opt_kws)
        elif not hasattr(self, 'estimator_'):
            self.estimator_ = clone(self.estimator)
            # we will check later that the estimator is properly fitted
        return self

    def _fit_dr_transformer(self, X):
        """Fit the transformer with X and calculate subspace_var_ and
        subspace_var_ratio_

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns self.
        """
        check_is_fitted(self, 'estimator_')
        grad = self._get_estimator_gradients(X)
        self.dr_transformer_ = clone(self.dr_transformer)
        self.dr_transformer_.fit(grad)
        self._check_transformer(self.dr_transformer_)
        self.components_ = deepcopy(self.dr_transformer_.components_)
        comps, var_, var_ratio_ = subspace_variance(grad, self.components_.T)
        self.components_ = comps.T
        self.subspace_variance_ = var_
        self.subspace_variance_ratio_ = var_ratio_
        return self

    def get_estimator_gradients(self, X):
        """Returns gradients of the sample

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data for which gradients will be calculated.

        Returns
        -------
        grad : ndarray, shape (n_samples, n_features)
            Calculated gradients.
        """
        X = check_array(X)
        return self._get_estimator_gradients(X)

    def _get_estimator_gradients(self, X):
        """Returns gradients of the sample and check model is fitted

        See ``get_estimator_gradients``
        """
        check_is_fitted(self, 'estimator_')
        grad = self.estimator_.predict_gradient(X)
        return grad

    def transform(self, X):
        """Apply dimensionality reduction on X.

        X is projected on the components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : ndarray, shape (n_samples, n_components)
        """
        check_is_fitted(self, 'components_')
        X = check_array(X)
        return np.dot(X, self.components_.T)

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples in the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original : ndarray, shape (n_samples, n_features)
        """
        check_is_fitted(self, 'components_')
        X = check_array(X)
        return np.dot(X, np.linalg.pinv(self.components_).T)

    @property
    def feature_importances_(self):
        """Return the feature importances of each feature in new axes.

        Returns
        -------
        feature_importances_ : ndarray, shape (n_components, n_features)
            The contribution of each feature to new axes.
            n_components = ``dr_transformer.n_components``
        """
        check_is_fitted(self, 'components_')
        return self.components_
