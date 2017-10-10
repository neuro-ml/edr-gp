import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
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
        check_is_fitted(self.estimator, 'estimator_')

    def _check_dr_transformer(self):
        if not hasattr(self.dr_transformer, 'components_'):
            raise AttributeError('The transformer does not expose '
                                 '"components_" attribute')

    def fit(self, X, y=None, **opt_kws):
        if y is None:
            self._check_estimator_fitted()
        else:
            self.estimator.fit(X, y, **opt_kws)

        grad = self.estimator.predict_gradient(X)
        self.dr_transformer.fit(grad)
        self._check_dr_transformer()
        self._set_components_(self.dr_transformer.components_)
        return self

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


class EffectiveDimensionalityReduction(_BaseEDR):

    def __init__(self, estimator, dr_transformer, normalize=True,
                 preprocess_by_pca=False, **pca_kwargs):
        self.normalize = normalize
        self.preprocess_by_pca = preprocess_by_pca
        self.pca_kwargs = pca_kwargs
        super(EffectiveDimensionalityReduction, self).__init__(
            estimator, dr_transformer)

    def fit(self, X, y=None, **opt_kws):
        X = self._fit_preprocessing(X)
        super(EffectiveDimensionalityReduction, self).fit(X, y, **opt_kws)
        return self

    def _fit_preprocessing(self, X, transform=True):
        if not self.normalize:
            if self.preprocess_by_pca:
                mes = 'To apply PCA prerpocessing, normalize should be True'
                raise ValueError(mes)
            return X
        sc = StandardScaler()
        X_preprocessed = sc.fit_transform(X)
        # initialize self.components_
        self.components_ = np.diag(1 / sc.scale_)
        # note that X will be centered during training to improve
        # robustness of GP models.
        # the transform step will be a pure linear map without a translation

        if self.preprocess_by_pca:
            pca = PCA(**self.pca_kwargs)
            X_preprocessed = pca.fit(X_preprocessed)
            # update self.components_
            self._set_components_(pca.components_)
        return X_preprocessed if transform else None

    def _set_components_(self, components):
        self.components_ = (
            deepcopy(components) if not hasattr(self, 'components_')
            else np.dot(components, self.components_))
