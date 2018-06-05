"""Base class for Effective Dimensionality Reduction"""

import numpy as np
import warnings
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize
from .utils import subspace_variance_ratio


class BaseEDR(BaseEstimator, TransformerMixin):
    """Base class for effective dimensionality reduction

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``predict_gradient``
        and ``fit`` methods.
    transformer : objec
        A linear dimensionnality reduction method that provides
        information about new axes through ``components_`` attribute.
    n_components : int (default=None)
        Number of components to left after fitting. If None then 
        n_components = n_features

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        New axes in feature space, representing the directions of
        maximum variance of the target.
        n_components = ``transformer.n_components``
    feature_importances_ : ndarray, shape (n_components, n_features)
        The contribution of each feature to new axes.
        n_components = ``transformer.n_components``
    estimator_ : object
        Estimator fitted to preprocessed data.
    transformer_ : object
        `transformer` fitted to gradients.
    subspace_variance_: array, shape (n_components, )
        Subspace variance calculated as tr(X.T * X) - tr(Y_i.T * Y_i)
        where Y_i=XU_i, U_i - orthogonal complement for components_.T[:, :i],
        i =  1, ..., n_components
        n_components = ``transformer.n_components``
    subspace_variance_ratio_: array, (n_components, )
        Subspace variance ratio calculated as subspace_var_/tr(X.T * X)
        n_components = ``transformer.n_components``

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

    def __init__(self, estimator=None, transformer=None, n_components=None,
                 step=None):
        self.estimator = estimator
        self.transformer = transformer
        self.n_components = n_components

    def _check_init(self, n_features):
        if self.estimator is None:
            raise ValueError("Estimator should be speciified")

        if self.transformer is None:
            raise ValueError("transformer should be specified")

        if self.n_components is None:
            self.n_components_ = n_features
        else:
            self.n_components_ = self.n_components

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
        self._check_init(X.shape[1])
        self.components_ = None
        self.num_iter = 0
        self._fit_estimator(X, y, **opt_kws)
        self._fit_transformer(X)
        X_proj = self.transform(X)
        self.num_iter += 1
        self._last_fit(X_proj, y, **opt_kws)
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
            if self.num_iter == 0:
                self.first_estimator_ = clone(self.estimator_)
        elif not hasattr(self, 'estimator_'):
            self.estimator_ = clone(self.estimator)
            # we will check later that the estimator is properly fitted
        return self

    def _fit_transformer(self, X):
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
        if self.num_iter == 0:
            self._first_gradients_ = grad

        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(grad)
        self._check_transformer(self.transformer_)
        components = deepcopy(self.transformer_.components_)
        components = components[:self.n_components_, :]
        self.components_ = (components if self.components_ is None else
                            np.dot(components, self.components_))
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
        self.subspace_gradients_ = self._get_estimator_gradients(X)
        self._recovered_gradients_ = np.dot(self.subspace_gradients_,
                                            self.components_)
        (self.subspace_variance_,
         self.subspace_variance_ratio_) = subspace_variance_ratio(
            self._first_gradients_,
            self.components_.T)
        return self

    def refit(self, refit_transformer, index=None):
        """Compute new components using gradients estimated during fit.

        It uses gradients estimated during fit and finds the right subspace 
        for them using `refit_transformer`. To transform data with 
        found components use `transform` with `refitted=True`

        Parameters
        ----------
        refit_transformer : object
            Transformer to fit with gradients estimated on fit.
            It should have attribute `components_` after fit.

        index : array-like
            Indices of objects for which transform will be applied.

        Returns
        -------
        self : object
            Returns self.
        """
        check_is_fitted(self, 'components_')
        if index is None:
            index = slice(None)
        self.refit_transformer_ = clone(refit_transformer)
        self.refit_transformer_.fit(self._first_gradients_[index, :])
        self._check_transformer(self.refit_transformer_)
        self.refit_components_ = deepcopy(self.refit_transformer_.components_)
        self.refit_components_ = normalize(self.refit_components_, axis=1)

        self.refit_components_ = self._remove_zero_components(
            self.refit_components_)

        (self.refit_subspace_variance_,
         self.refit_subspace_variance_ratio_) = subspace_variance_ratio(
            self._first_gradients_[index, :],
            self.refit_components_.T)
        return self

    def _remove_zero_components(self, components):
        nonzero_indices = np.nonzero(np.linalg.norm(components, axis=1))[0]
        zero_components = list(
            set(range(components.shape[0])) - set(nonzero_indices))
        if zero_components:
            mes = ('Components with numbers {} will be droped because they '
                   'contains only zeros').format(zero_components)
            warnings.warn(mes, RuntimeWarning)
        components = np.delete(components, zero_components, axis=0)
        return components

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
            n_components = ``transformer.n_components``
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


class IterativeEDR(BaseEDR):
    """Class for iterative effective dimensionality reduction

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``predict_gradient``
        and ``fit`` methods.
    transformer : object
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
        n_components = ``transformer.n_components``
    feature_importances_ : ndarray, shape (n_components, n_features)
        The contribution of each feature to new axes.
        n_components = ``transformer.n_components``
    estimator_ : object
        Estimator fitted to preprocessed data.
    transformer_ : object
        `transformer` fitted to gradients.
    subspace_variance_: array, shape (n_components, )
        Subspace variance calculated as tr(X.T * X) - tr(Y_i.T * Y_i)
        where Y_i=XU_i, U_i - orthogonal complement for components_.T[:, :i],
        i =  1, ..., n_components
        n_components = ``transformer.n_components``
    subspace_variance_ratio_: array, (n_components, )
        Subspace variance ratio calculated as subspace_var_/tr(X.T * X)
        n_components = ``transformer.n_components``

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

    def __init__(self, estimator=None, transformer=None, n_components=None,
                 step=None):
        self.estimator = estimator
        self.transformer = transformer
        self.n_components = n_components
        self.step = step

    def _check_step(self, n_features):
        self.adaptive_step = False
        if self.step is None:
            self.step_ = self.n_components_
        elif isinstance(self.step, int) and self.step > 0:
            if self.n_components_ == n_features:
                mes = "If step is int (n_components < n_features) must be True"
                raise ValueError(mes)
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
        n_features = X.shape[1]
        self._check_init(n_features)
        self._check_step(n_features)

        self.components_ = None
        self.continue_iteration = True
        self.num_iter = 0
        X_proj = X.copy()
        while self.continue_iteration:
            self._fit_estimator(X_proj, y, **opt_kws)
            self._fit_transformer(X_proj)
            X_proj = self.transform(X)
            self.num_iter += 1

        self._last_fit(X_proj, y, **opt_kws)
        return self

    def _fit_transformer(self, X):
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
        if self.num_iter == 0:
            self._first_gradients_ = grad

        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(grad)
        self._check_transformer(self.transformer_)
        comps = deepcopy(self.transformer_.components_)

        n_components = self._select_n_components(grad, comps)
        self.components_ = self._select_best_components(comps, n_components)
        return self

    def _select_n_components(self, grad, components):
        if self.adaptive_step:
            _, var_ratio_ = subspace_variance_ratio(grad, components.T)
            n_components = np.sum(np.cumsum(var_ratio_) < self.step_,
                                  dtype=int) + 1
            if n_components == grad.shape[1]:
                self.continue_iteration = False
        else:
            n_components = max(self.n_components_, grad.shape[1] - self.step_)
            if n_components == self.n_components_:
                self.continue_iteration = False
        return n_components

    def _select_best_components(self, components, n_components):
        self.components_ = (components if self.components_ is None else
                            np.dot(components, self.components_))
        _, var_ratio = subspace_variance_ratio(self._first_gradients_,
                                               self.components_.T)
        best_components = np.argsort(var_ratio)[-n_components:][::-1]
        return self.components_[best_components, :]


class BlockEDR(BaseEDR):
    """Class for block effective dimensionality reduction

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``predict_gradient``
        and ``fit`` methods.
    transformer : object
        A linear dimensionnality reduction method that provides
        information about new axes through ``components_`` attribute.
    n_components : int (default=None)
        Number of components to left after fitting. If None then 
        n_components = n_features
    blocks : object
        List of lists. i-th list should specify column indices for 
        i-th block.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        New axes in feature space, representing the directions of
        maximum variance of the target.
        n_components = ``transformer.n_components``
    feature_importances_ : ndarray, shape (n_components, n_features)
        The contribution of each feature to new axes.
        n_components = ``transformer.n_components``
    estimator_ : object
        Estimator fitted to preprocessed data.
    transformer_ : object
        `transformer` fitted to gradients.
    subspace_variance_: array, shape (n_components, )
        Subspace variance calculated as tr(X.T * X) - tr(Y_i.T * Y_i)
        where Y_i=XU_i, U_i - orthogonal complement for components_.T[:, :i],
        i =  1, ..., n_components
        n_components = ``transformer.n_components``
    subspace_variance_ratio_: array, (n_components, )
        Subspace variance ratio calculated as subspace_var_/tr(X.T * X)
        n_components = ``transformer.n_components``

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

    def __init__(self, estimator, transformer, n_components=None, blocks=None):
        self.estimator = estimator
        self.transformer = transformer
        self.blocks = blocks
        self.n_components = n_components

    def _fit_transformer(self, X):
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
        n_features = X.shape[1]
        self._make_blocks(n_features)

        grad = self._get_estimator_gradients(X)
        if self.num_iter == 0:
            self._first_gradients_ = grad

        components = []
        for block in self.blocks_:
            transformer = clone(self.transformer)
            components.append(self._fit_single_block(transformer, grad, block))

        self.components_ = self._merge_components(components)
        return self

    def _fit_single_block(self, transformer, grads, block, index=None,
                          params=None):
        """Fit the transformet with gradients of block

        Parameters
        ----------
        transformer : object
            A linear dimensionnality reduction method that provides
            information about new axes through ``components_`` attribute.
        grads : array-like, shape (n_samples, n_features_in_block)
            Gradients of `estimator` that correspond to block columns
        block : object
            Dict that contains information about block.
        index : array-like
            Indices that specify which object should be accounted 
            for fitting transformer
        params : dict
            Additional kwargs for transformer

        Returns
        -------
        components_T : array-like
            Projector matrix for the block
        """
        columns = block['columns']
        if index is None:
            index = np.matlib.repmat(True, 1, grads.shape[0])[0]
        grads_block = grads[np.ix_(index, columns)]
        transformer.set_params(n_components=block['n_components'])
        if params is not None:
            transformer.set_params(**params)

        transformer.fit(grads_block)

        return transformer.components_.T

    def _merge_components(self, components):
        """Merges components for different block in projector matrix

        Parameters
        ----------
        components : list
            List of projector matrices for all blocks

        Returns
        -------
        components : array-like
            Merged projector matrix
        """
        n_features = self._first_gradients_.shape[1]

        eff_dim = sum(components[i].shape[1] for i in range(len(components)))
        ndim = n_features
        result = np.zeros((ndim, eff_dim))
        start = 0
        stop = 0
        for i, component in enumerate(components):
            stop += component.shape[1]
            result[self.blocks_[i]['columns'], start:stop] = component
            self.blocks_[i]['columns'] = np.arange(start, stop)
            start += component.shape[1]

        return result.T

    def refit(self, refit_transformer, index=None, params=None):
        """Compute new components using gradients estimated during fit.

        It uses gradients estimated during fit and finds the right subspace 
        for them using `refit_transformer`. To transform data with 
        found components use `transform` with `refitted=True`

        Parameters
        ----------
        refit_transformer : object
            Transformer to fit with gradients estimated on fit.
            It should have attribute `components_` after fit.

        index : array-like
            Indices of objects for which transform will be applied.

        params : list of dict
            List of dictionaries containing additional kwargs for 
            transfomer. i-th dict should contain kwargs for transformer
            that will be fitted to i-th block of features.

        Returns
        -------
        self : object
            Returns self.
        """
        n_features = self._first_gradients_.shape[1]
        self._make_blocks(n_features)
        self.refit_transformer_ = clone(refit_transformer)
        if index is None:
            self.refit_index_ = np.matlib.repmat(
                True, 1, self._first_gradients_.shape[0])[0]
        else:
            self.refit_index_ = index

        components = []
        for block_num, block in enumerate(self.blocks_):
            components.append(self._fit_single_block(
                refit_transformer, self._first_gradients_, block,
                index=self.refit_index_,
                params=None if params is None else params[block_num]))

        self.refit_components_ = normalize(self._merge_components(components))
        self.refit_components_ = self._remove_zero_components(
            self.refit_components_)

        (self.refit_subspace_variance_,
         self.refit_subspace_variance_ratio_) = subspace_variance_ratio(
            self._first_gradients_[self.refit_index_, :],
            self.refit_components_.T)

        return self

    def _make_blocks(self, n_features):
        """Make the right structure of blocks for futher processing.
        """

        if self.blocks is None:
            if isinstance(self.n_components_, int):
                self.blocks_ = [
                    {
                        'columns': np.arange(n_features),
                        'n_components': self.n_components_,
                    }
                ]
            else:
                mes = "blocks should be specified if n_components is list"
                raise ValueError(mes)
        elif isinstance(self.blocks, list):
            if isinstance(self.n_components_, list):
                self.blocks_ = [
                    {
                        'columns': block,
                        'n_components': n_comps,
                    } for block, n_comps in zip(self.blocks,
                                                self.n_components_)
                ]
            elif isinstance(self.n_components_, int):
                self.blocks_ = [
                    {
                        'columns': block,
                        'n_components': max(self.n_components, len(block)),
                    } for block in self.blocks
                ]
        return self
