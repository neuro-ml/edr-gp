"""Effective Dimensionality Reduction"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, check_X_y
from sklearn.base import TransformerMixin, clone
from copy import deepcopy
from sklearn.utils.validation import check_is_fitted
from .utils import subspace_variance


class BaseEDR(TransformerMixin):
    """Base class for effective dimensionality reduction

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``predict_gradient``
        and ``fit`` methods.
    dr_transformer : object
        A linear dimensionnality reduction method that provides
        information about new axes through ``components_`` attribute.

    Attributes
    ----------
    components_ : array, shape (dr_transformer.n_components, n_features)
        New axes in feature space, representing the directions of
        maximum variance of the target.
    feature_importances_ : ndarray, shape (n_components, n_features)
        The contribution of each feature to new axes.
        n_components = ``dr_transformer.n_components``
    estimator_ : object
        Estimator fitted to preprocessed data.
    dr_transformer_ : object
        Dr_transformer fitted to gradients.
    subspace_var_: array, shape (dr_transformer.n_components, )
        Subspace variance calculated as tr(X.T * X) - tr(Y_i.T * Y_i)
        where Y_i=XU_i, U_i - orthogonal complement for components_.T[:, :i],
        i =  1, ..., dr_transformer.n_components
    subspace_var_ratio_: array, (dr_transformer.n_components, )
        Subspace variance ratio calculated as subspace_var_/tr(X.T * X)
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

    def fit(self, X, y=None, method='optimize', **opt_kws):
        """Fit the model with X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        method : {'optimize', 'optimize_restarts'}, optional
            Invokes passed method to fit `estimator`.
            For 'optimize_restarts' perform random restarts of the
            model, and set the model to the best.

        Returns
        -------
        self : object
            Returns self.
        """
        self._fit_estimator(X, y, method, **opt_kws)
        self._fit_dr_transformer(X)
        return self

    def _fit_estimator(self, X, y=None, method='optimize', **opt_kws):
        """Fit the estimator with X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        method : {'optimize', 'optimize_restarts'}, optional
            Invokes passed method to fit `estimator`.
            For 'optimize_restarts' perform random restarts of the
            model, and set the model to the best.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        if y is not None:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, method, **opt_kws)
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


class EffectiveDimensionalityReduction(BaseEDR):
    """Effective dimensionality reduction class

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``predict_gradient``
        and ``fit`` methods.
    dr_transformer : object
        A linear dimensionnality reduction method that provides
        information about new axes through ``components_`` attribute.
    normalize : bool, optional (default=True)
        If True input data will be normalized by ``StandardScaler``
        before fitting.
    preprocessor : object, optional (default=None)
        Performs preprocessing of input data if it was normalized.
        If None preprocessing won't be done. Use PCA, SparsePCA etc.

    Attributes
    ----------
    components_ : array, shape (dr_transformer.n_components, n_features)
        New axes in feature space, representing the directions of
        maximum variance of the target.
    feature_importances_ : ndarray
        ???
    estimator_ : object
        Estimator fitted to preprocessed data.
    dr_transformer_ : object
        Dr_transformer fitted to gradients.
    scaler_ : object
        ``StandardScaler`` fitted to raw data.
    preprocessor_ : object
        Preprocessor fitted to normalized data.
    subspace_var_: array, shape (dr_transformer.n_components, )
        Subspace variance calculated as tr(X.T * X) - tr(Y_i.T * Y_i)
        where Y_i=XU_i, U_i - orthogonal complement for components_.T[:, :i],
        i =  1, ..., dr_transformer.n_components
    subspace_var_ratio_: array, (dr_transformer.n_components, )
        Subspace variance ratio calculated as subspace_var_/tr(X.T * X)
    """

    def __init__(self, estimator, dr_transformer, normalize=True,
                 preprocessor=None):
        self.normalize = normalize
        self.preprocessor = preprocessor
        super(EffectiveDimensionalityReduction, self).__init__(
            estimator, dr_transformer)

    def fit(self, X, y=None, method='optimize', **opt_kws):
        """Fit the model with X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        method : {'optimize', 'optimize_restarts'}, optional
            Invokes passed method to fit `estimator`.
            For 'optimize_restarts' perform random restarts of the
            model, and set the model to the best.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._preprocessing_fit(X)
        super(EffectiveDimensionalityReduction,
              self).fit(X, y, method, **opt_kws)
        if self.normalize is True:
            self.components_ = np.dot(self.components_, self._scaling_)
        return self

    def _preprocessing_fit(self, X, transform=True):
        """Preprocess X with ``StandardScaler`` and `preprocessor`

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        transform : bool, optional (default=True)

        Returns
        -------
        X_new : ndarray, shape (n_samples, n_componets)
        """
        if not self.normalize:
            if self.preprocessor is not None:
                mes = 'To apply prerpocessing, normalize should be True'
                raise ValueError(mes)
            return X
        self.scaler_ = StandardScaler()
        X_preprocessed = self.scaler_.fit_transform(X)
        self._scaling_ = np.diag(self.scaler_.scale_)
        self._reverse_scaling_ = np.diag(1 / self.scaler_.scale_)
        # note that X will be centered during training to improve
        # robustness of GP models.
        # the transform step will be a pure linear map without a translation

        if self.preprocessor is not None:
            self.preprocessor_ = clone(self.preprocessor)
            X_preprocessed = self.preprocessor_.fit_transform(X_preprocessed)
            self._check_transformer(self.preprocessor_)
            # save preprocesing map
            self._preprocessing_ = self.preprocessor_.components_
        return X_preprocessed if transform else None

    def _preprocessing_transform(self, X):
        """ """
        X = check_array(X)
        if self.normalize is True:
            check_is_fitted(self, 'scaler_')
            X = self.scaler_.transform(X)
        if self.preprocessor is not None:
            check_is_fitted(self, 'preprocessor_')
            X = self.preprocessor_.transform(X)
        return X

    def get_estimator_gradients(self, X):
        X = check_array(X)
        return self._get_estimator_gradients(X, True)

    def _get_estimator_gradients(self, X, prepocess=False):
        """Returns gradients of the sample

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data for which gradients will be calculated.
        preprocess : bool, optional (default=False)

        Returns
        -------
        grad : ndarray, shape (n_samples, n_features)
            Calculated gradients in the non-preprocessed space.
        """
        if prepocess:
            X = self._preprocessing_transform(X)
        check_is_fitted(self, 'estimator_')
        grad = self.estimator_.predict_gradient(X)
        if self.preprocessor is not None:
            check_is_fitted(self, 'preprocessor_')
            grad = np.dot(grad, self._preprocessing_)
        return grad

    @property
    def feature_importances_(self):
        """Return the feature importances of each feature in new axes.

        Feature importances are calculated with respect to scaling.

        Returns
        -------
        feature_importances_ : ndarray, shape (n_components, n_features)
            The contribution of each feature to new axes.
            n_components = ``dr_transformer.n_components``
        """
        check_is_fitted(self, 'components_')
        importances_ = self.components_
        if self.normalize is True:
            importances_ = np.dot(importances_, self._reverse_scaling_)
        return importances_
