import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, check_X_y
from sklearn.base import TransformerMixin, clone
from copy import deepcopy
from sklearn.utils.validation import check_is_fitted


class _BaseEDR(TransformerMixin):
    """Base class for Effective Dimensionality Reduction.
    It performs a single step of dimensionality reduction.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(self, estimator, dr_transformer):
        self.estimator = estimator
        self.dr_transformer = dr_transformer

    def _check_estimator_fitted(self):
        check_is_fitted(self, 'estimator_')
        check_is_fitted(self.estimator_, 'estimator_')

    def _check_transformer(self, transformer):
        if not hasattr(transformer, 'components_'):
            raise AttributeError('The transformer does not expose '
                                 '"components_" attribute')

    def fit(self, X, y=None, **opt_kws):
        X, y = check_X_y(X, y, accept_sparse=False)
        if y is not None:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **opt_kws)
        grad = self.get_estimator_gradients(X)
        self.dr_transformer_ = clone(self.dr_transformer)
        self.dr_transformer_.fit(grad)
        self._check_transformer(self.dr_transformer_)
        self._set_components_(self.dr_transformer_.components_)
        return self

    def get_estimator_gradients(self, X):
        self._check_estimator_fitted()
        X = check_array(X)
        grad = self.estimator_.predict_gradient(X)
        return grad

    def transform(self, X):
        check_is_fitted(self, 'components_')
        X = check_array(X)
        return np.dot(X, self.components_.T)

    def inverse_transform(self, X):
        check_is_fitted(self, 'components_')
        X = check_array(X)
        return np.dot(X, np.linalg.pinv(self.components_).T)

    def _set_components_(self, components):
        self.components_ = deepcopy(components)

    @property
    def feature_importances_(self):
        check_is_fitted(self, 'components_')
        return self.components_


class EffectiveDimensionalityReduction(_BaseEDR):

    def __init__(self, estimator, dr_transformer, normalize=True,
                 preprocessor=None):
        self.normalize = normalize
        self.preprocessor = preprocessor
        super(EffectiveDimensionalityReduction, self).__init__(
            estimator, dr_transformer)

    def fit(self, X, y=None, **opt_kws):
        X = self._fit_preprocessing(X)
        super(EffectiveDimensionalityReduction, self).fit(X, y, **opt_kws)
        return self

    def _fit_preprocessing(self, X, transform=True):
        if not self.normalize:
            if self.preprocessor is not None:
                mes = 'To apply prerpocessing, normalize should be True'
                raise ValueError(mes)
            return X
        sc = StandardScaler()
        X_preprocessed = sc.fit_transform(X)
        # initialize self.components_
        self.components_ = np.diag(1 / sc.scale_)
        self._reverse_scaling_ = np.diag(sc.scale_)
        # note that X will be centered during training to improve
        # robustness of GP models.
        # the transform step will be a pure linear map without a translation

        if self.preprocessor is not None:
            self.preprocessor_ = clone(self.preprocessor)
            X_preprocessed = self.preprocessor_.fit_transform(X_preprocessed)
            self._check_transformer(self.preprocessor_)
            # update self.components_
            self._set_components_(self.preprocessor_.components_)
        return X_preprocessed if transform else None

    def _set_components_(self, components):
        self.components_ = (
            deepcopy(components) if not hasattr(self, 'components_')
            else np.dot(components, self.components_))

    @property
    def feature_importances_(self):
        check_is_fitted(self, 'components_')
        return np.dot(self.components_, self._reverse_scaling_)
