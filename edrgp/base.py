"""Base class for Effective Dimensionality Reduction"""

import numpy as np
from copy import deepcopy
from sklearn.base import TransformerMixin, clone
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from .utils import subspace_variance_ratio


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
    n_components : int (default=None)
        Number of components to left after fitting. If None then 
        n_components = n_features
    step : int, float (default=None)
        Number of components to drop at each iteration. If step is float
        then number of components to drop at each iteration defines as 
        number of worst components with sum of subspace variance lower then 
        1 - step. If step is None only one iteration is applied.

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

    def __init__(self, estimator, dr_transformer, n_components=None,
                 step=None):
        self.estimator = estimator
        self.dr_transformer = dr_transformer
        self.n_components = n_components
        self.step = step

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

        if self.n_components is None:
            self.n_components_ = X.shape[1]
            if isinstance(self.step, int):
                mes = "If step is int n_components must be specified"
                raise ValueError(mes)
        else:
            self.n_components_ = self.n_components

        self.adaptive_step = False
        if self.step is None:
            self.step_ = self.n_components_
        elif isinstance(self.step, int) and self.step > 0:
            self.step_ = self.step
        elif isinstance(self.step, float) and 0 < self.step < 1:
            if self.n_components is not None:
                mes = "If step is float n_components should be None"
                raise ValueError(mes)
            self.adaptive_step = True
            self.step_ = self.step
        else:
            mes = "Step should be None or int > 0 or float from 0 to 1"
            raise ValueError(mes)

        self._gradients_ = None
        self.components_ = None
        self.continue_iteration = True
        self.num_iter = 0
        X_proj = X.copy()
        while self.continue_iteration:
            self._fit_estimator(X_proj, y, **opt_kws)
            self._fit_dr_transformer(X_proj)
            X_proj = self.transform(X)
            self.num_iter += 1

        self._last_fit(X_proj, y, **opt_kws)
        return self

    def refit(self, refit_transformer):
        """Compute new components using gradients estimated during fit.

        It uses gradients estimated during fit and finds the right subspace 
        for them using `refit_transformer`. To transform data with 
        found components use `transform` with `refiitted=True`

        Parameters
        ----------
        refit_transformer : object
            Transformer to fit with gradients estimated on fit.
            It should have attribute `components_` after fit.

        Returns
        -------
        self : object
            Returns self.
        """
        check_is_fitted(self, 'components_')
        self.refit_transformer_ = clone(refit_transformer)
        self.refit_transformer_.fit(self._gradients_)
        self._check_transformer(self.refit_transformer_)
        self.refit_components_ = deepcopy(self.refit_transformer_.components_)
        self.refit_components_ = (
            self.refit_components_/np.linalg.norm(self.refit_components_,
                                                  axis=1).reshape(-1, 1))
        (self.refit_subspace_variance_,
         self.refit_subspace_variance_ratio_) = subspace_variance_ratio(
            self._gradients_,
            self.refit_components_.T)
        return self

    def _last_fit(self, X, y, **opt_kws):
        """Compute gradients for original and effective subspaces.

        Also computes subspace variance ratio for gradients in effective 
        subspace with respect to original gradients

        Parameters
        ----------
        X : array-like, shape (n_samples, n_componets)
            Training data, projected to effective subspace.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
                Returns self.

        """
        self._fit_estimator(X, y, **opt_kws)
        check_is_fitted(self, 'estimator_')
        grad = self._get_estimator_gradients(X)
        self.subspace_gradients_ = grad
        self._gradients_ = np.dot(grad, self.components_)
        (self.subspace_variance_,
         self.subspace_variance_ratio_) = subspace_variance_ratio(
            self._gradients_,
            self.components_.T)
        return self

    def _fit_estimator(self, X, y, **opt_kws):
        """Fit the estimator with X, y.

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
        components = deepcopy(self.dr_transformer_.components_)

        if self.adaptive_step:
            _, var_ratio_ = subspace_variance_ratio(grad, components.T)
            n_components = np.sum(np.cumsum(var_ratio_) < self.step_,
                                  dtype=int) + 1
            if n_components == grad.shape[1]:
                self.continue_iteration = False
        else:
            n_components = max(self.n_components_, X.shape[1] - self.step_)
            if n_components == self.n_components_:
                self.continue_iteration = False

        components = components[:n_components, :]
        self.components_ = (components if self.components_ is None else
                            np.dot(components, self.components_))
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

    def transform(self, X, refitted=False):
        """Apply dimensionality reduction on X.

        X is projected on the components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        refitted : bool (False)
            Whether to transform using refit_transformer components.
            May be set to `True` only if `refit` was applied before
        Returns
        -------
        X_new : ndarray, shape (n_samples, n_components)
        """
        check_is_fitted(self, 'components_')
        X = check_array(X)
        if refitted:
            check_is_fitted(self, ['refit_transformer_', 'refit_components_'])
            return np.dot(X, self.refit_components_.T)
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

    def _check_transformer(self, transformer):
        """Check that transformer has attribute ``components_``.

        Parameters
        ----------
        transformer : object
        """
        if not hasattr(transformer, 'components_'):
            raise AttributeError('The transformer does not expose '
                                 '"components_" attribute')


class ExtendedEDR(BaseEDR):

    def __init__(self, estimator, dr_transformer, refit_transformer,
                 n_components=None, step=None):
        super(ExtendedEDR, self).__init__(estimator, dr_transformer,
                                          n_components, step)
        self.refit_transformer = refit_transformer

    def fit(self, X, y=None, **opt_kws):
        super(ExtendedEDR, self).fit(X, y, **opt_kws)
        self.refit(self.refit_transformer)

    def transform(self, X):
        super(ExtendedEDR, self).transform(X, refitted=True)
