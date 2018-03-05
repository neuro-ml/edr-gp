"""Effective Dimensionality Reduction"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from .base import BaseEDR


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
    n_components : int (default=None)
        Number of components to left after fitting. If None then 
        n_components = n_features
    step : int, float (default=None)
        Number of components to drop at each iteration. If step is float
        then number of components to drop at each iteration defines as 
        number of worst components with sum of subspace variance lower then 
        1 - step. If step is None only one iteration is applied.
    normalize : bool, optional (default=True)
        If True input data will be normalized by ``StandardScaler``
        before fitting.
    preprocessor : object, optional (default=None)
        Performs preprocessing of input data if it was normalized.
        If None preprocessing won't be done. Use PCA, SparsePCA etc.

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
    scaler_ : object
        ``StandardScaler`` fitted to raw data.
    preprocessor_ : object
        Preprocessor fitted to normalized data.
    subspace_variance_: array, shape (n_components, )
        Subspace variance calculated as tr(X.T * X) - tr(Y_i.T * Y_i)
        where Y_i=XU_i, U_i - orthogonal complement for components_.T[:, :i],
        i =  1, ..., n_components
        n_components = ``dr_transformer.n_components``
    subspace_variance_ratio_: array, (n_components, )
        Subspace variance ratio calculated as subspace_var_/tr(X.T * X)
        n_components = ``dr_transformer.n_components``

    If `refit` method has been applied the following attributes
    are also presented:

    refit_transformer_ : object
        `refit_transformer` fitted on gradients estimated during `fit`.
        Attribute is present only if the `refit` has been applied.
    refit_components_ : array, shape (n_refit_components, n_features)
        New axes in feature space, representing the directions of
        maximum variance of the target.
        n_components = ``refit_transformer.n_components``
    refit_subspace_variance_: array, shape (n_components, )
        Subspace variance calculated as tr(X.T * X) - tr(Y_i.T * Y_i)
        where Y_i=XU_i, U_i - orthogonal complement for components_.T[:, :i],
        i =  1, ..., n_components
        n_components = ``refit_transformer.n_components``
    refit_subspace_variance_ratio_: array, (n_components, )
        Subspace variance ratio calculated as subspace_var_/tr(X.T * X)
        n_components = ``refit_transformer.n_components``
    """

    def __init__(self, estimator, dr_transformer, n_components=None, step=None,
                 normalize=True, preprocessor=None):
        self.normalize = normalize
        self.preprocessor = preprocessor
        super(EffectiveDimensionalityReduction, self).__init__(
            estimator, dr_transformer, n_components, step)

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
        X = self._preprocessing_fit(X)
        super(EffectiveDimensionalityReduction,
              self).fit(X, y, **opt_kws)
        if self.normalize is True:
            self.components_ = np.dot(self.components_, self._reverse_scaling_)
            if hasattr(self, 'refit_components_'):
                self.refit_components_ = np.dot(self.refit_components_,
                                                self._reverse_scaling_)
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
        """Preprocess new data

        Applies preprocessing steps as to data in ``fit``

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_new : ndarray, shape (n_samples, n_componets)
        """
        X = check_array(X)
        if self.normalize is True:
            check_is_fitted(self, 'scaler_')
            X = self.scaler_.transform(X)
        # if self.preprocessor is not None:
        #     check_is_fitted(self, 'preprocessor_')
        #     X = self.preprocessor_.transform(X)
            X = np.dot(X, self._scaling_)
        X = np.dot(X, self.components_.T)
        return X

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
        # X_transform = self.transform(X)
        # subsp_grads = self.estimator_.predict_gradient(X_transform)
        return self._get_estimator_gradients(X, True)

    def _get_estimator_gradients(self, X, preprocessing_transform=False):
        """Returns gradients of the sample

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data for which gradients will be calculated.
        preprocess : bool, optional (default=False)
                Whether to preprocess data or not.

        Returns
        -------
        grad : ndarray, shape (n_samples, n_features)
            Calculated gradients in the non-preprocessed space.
        """
        if preprocessing_transform:
            X = self._preprocessing_transform(X)
        check_is_fitted(self, 'estimator_')
        grad = self.estimator_.predict_gradient(X)
        if (self.preprocessor is not None and
            self.num_iter == 0 and
                not preprocessing_transform):
            check_is_fitted(self, 'preprocessor_')
            grad = np.dot(grad, self._preprocessing_)
        if preprocessing_transform:
            grad = np.dot(grad, self.components_)
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
            importances_ = np.dot(importances_, self._scaling_)
        return importances_

    def transform(self, X, refitted=False):
        check_is_fitted(self, 'components_')
        X = check_array(X)
        if refitted:
            check_is_fitted(self, ['refit_transformer_', 'refit_components_'])
            return np.dot(X, self.refit_components_.T)
        if hasattr(self, '_gradients_') and self._gradients_ is not None:
            components = self.components_
        else:
            components = (self.components_ if self.preprocessor is None else
                          np.dot(self.components_, self._preprocessing_.T))
        return np.dot(X, components.T)
