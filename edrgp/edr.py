import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, check_X_y
from sklearn.base import TransformerMixin, clone
from copy import deepcopy
from sklearn.utils.validation import check_is_fitted


class BaseEDR(TransformerMixin):

    def __init__(self, estimator, dr_transformer):
        self.estimator = estimator
        self.dr_transformer = dr_transformer

    def _check_transformer(self, transformer):
        if not hasattr(transformer, 'components_'):
            raise AttributeError('The transformer does not expose '
                                 '"components_" attribute')

    def fit(self, X, y=None, **opt_kws):
        X, y = check_X_y(X, y, accept_sparse=False)
        if y is not None:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **opt_kws)
        elif not hasattr(self, 'estimator_'):
            self.estimator_ = clone(self.estimator)
            # we will check later that the estimator is properly fitted

        grad = self._get_estimator_gradients(X)
        self.dr_transformer_ = clone(self.dr_transformer)
        self.dr_transformer_.fit(grad)
        self._check_transformer(self.dr_transformer_)
        self.components_ = deepcopy(self.dr_transformer_.components_)
        return self

    def get_estimator_gradients(self, X):
        X = check_array(X)
        return self._get_estimator_gradients(X)

    def _get_estimator_gradients(self, X):
        check_is_fitted(self, 'estimator_')
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

    @property
    def feature_importances_(self):
        check_is_fitted(self, 'components_')
        return self.components_


class EffectiveDimensionalityReduction(BaseEDR):

    def __init__(self, estimator, dr_transformer, normalize=True,
                 preprocessor=None):
        self.normalize = normalize
        self.preprocessor = preprocessor
        super(EffectiveDimensionalityReduction, self).__init__(
            estimator, dr_transformer)

    def fit(self, X, y=None, **opt_kws):
        X = self._preprocessing_fit(X)
        super(EffectiveDimensionalityReduction, self).fit(X, y, **opt_kws)
        if self.normalize is True:
            self.components_ = np.dot(self.components_, self._scaling_)
        return self

    def _preprocessing_fit(self, X, transform=True):
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
        if prepocess:
            X = self._preprocessing_transform(X)
        check_is_fitted(self, 'estimator_')
        grad = self.estimator_.predict_gradient(X)
        if self.preprocessor is not None:
            check_is_fitted(self, 'preprocessor_')
            grad = np.dot(grad, self.preprocessor_.components_)
        return grad

    @property
    def feature_importances_(self):
        check_is_fitted(self, 'components_')
        importances_ = self.components_
        if self.normalize is True:
            importances_ = np.dot(importances_, self._reverse_scaling_)
        return importances_
