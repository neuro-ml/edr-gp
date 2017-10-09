import numpy as np
from sklearn.utils import check_X_y, assert_all_finite, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.decomposition import PCA
from copy import deepcopy
from GPy import kern as gpy_kern
from GPy.models import GPRegression, GPClassification


class BaseGP(BaseEstimator):
    
    def __init__(self, kernels=None, kernel_options=None, Y_metadata=None,
                 mean_function=None):
        self.kernels = kernels
        self.kernel_options = kernel_options
        self.Y_metadata = Y_metadata
        self.mean_function = mean_function

    def fit(self, X, y, **opt_kws):
        X, y = self._check_input(X, y)

        kernel = self._make_kernel(X)
        self.model_ = self._get_model(X, y, kernel)

        opt_kws.setdefault('messages', False)
        opt_kws.setdefault('max_iters', 1000)
        self.model_.optimize(**opt_kws)
        return self

    def _check_input(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        if self._estimator_type == 'classifier':
            check_classification_targets(y)
        y = y[:, np.newaxis]
        return X, y

    def _make_kernel(self, X):
        # kernel will be initiated as 'rbf' in model automatically
        if self.kernels is None:
            return self.kernels

        if isinstance(self.kernels, str):
            self.kernels = list(self.kernel)

        kernels = [getattr(gpy_kern, kern) for kern in self.kernels]
        input_dim = {'input_dim': X.shape[1]}

        if self.kernel_options is None:
            options = [input_dim] * len(self.kernels)
        elif len(kernels) == len(self.kernel_options):
            options = deepcopy(self.kernel_options)
            for opt in options:
                opt.update(input_dim)
        else:
            raise(ValueError)

        kernel = np.sum([kern(**opt) for kern, opt in zip(kernels, options)])
        return kernel

    def _get_model(self, X, y, kenrel):
        raise RuntimeError('Try to use GaussianProcessRegressor or'
                           'GaussianProcessClassifier')

    def predict(self, X):
        #         a.inverse_transform
        X = check_array(X, accept_sparse=False)
        y_pred = self.model_.predict(X)[0][:, 0]
        assert_all_finite(y_pred)
        return y_pred

    def predict_variance(self, X):
        return self.model_.predict(X)[1]

    def predict_gradient(self, X):
        #         print('UNSCALED GRADIENTS')
        return self.model_.predictive_gradients(X)[0][:, :, 0]
#     def predict_quantiles()


class GaussianProcessRegressor(BaseGP, RegressorMixin):

    def __init__(self, kernels=None, kernel_options=None, Y_metadata=None,
                 normalizer=None, noise_var=1.0, mean_function=None):
        self.normalizer = normalizer
        self.noise_var = noise_var

        super(GaussianProcessRegressor, self).__init__(kernels,
            kernel_options, Y_metadata, mean_function)

    def _get_model(self, X, y, kernel):
        return GPRegression(X, y, kernel, self.Y_metadata, self.normalizer,
                            self.noise_var, self.mean_function)


class GaussianProcessClassifier(BaseGP, ClassifierMixin):

    def _get_model(self, X, y, kernel):
        return GPClassification(X, y, kernel, self.Y_metadata,
                                self.mean_function)


class GaussianProcessesEDR(BaseGP):

    def __init__(self, estimator, method=None, n_components=None, step=None):
        self.estimator = estimator
        self.n_components = n_components
        self.step = step
        self.method = method

    def fit(self, X, y, **opt_kws):
        X, y = self._check_input(X, y)
        # Initialization
        n_features = X.shape[1]
        self.projection_ = None
        if self.n_components is None:
            n_components = n_features
        else:
            n_components = self.n_components

        if self.method is None:
            method = PCA()
        else:
            method = self.method
        _check_method(method)
        
        step = self.step

        if step is not None:
            X_projected = X.copy()
            for ndim in range(n_features, n_components+1, -step):
                self.single_fit(method, X_projected, y, ndim, **opt_kws)
                X_projected = self.project(X)
        self.single_fit(method, X, y, n_components, **opt_kws)

        self.model_ = self.estimator
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError('Fit the model before predicting')
        return self.model_.predict(self.project(X))

    def single_fit(self, method, X, y, ndim, **opt_kws):
        method.n_components = ndim
        self.estimator.fit(X, y, **opt_kws)
        grad = self.estimator.predict_gradient(X)
        method.fit(grad)

        if hasattr(method, 'components_'):
            components = method.components_.T
        else:
            raise AttributeError('The method {} does not expose '
                                 '"components_" attribute'
                                 'after fit'.format(method.__name__))

        self.projection_ = (components if self.projection_ is None else
                            np.dot(self.projection_, components))

    def project(self, X):
        if self.projection_ is None:
            raise RuntimeError('Fit the model before projection')
        return np.dot(X, self.projection_)
    
    def _check_method(method):
        if not hasattr(method, 'n_components'):
            raise AttributeError('The classifier does not expose '
                                 '"n_components" attribute')
        
