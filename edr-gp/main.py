import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA  # , SparsePCA
from copy import deepcopy
import GPy as gpy


class DummyModel(BaseEstimator, RegressorMixin):
    def __init__(self, model=None):
        self.model = model

    def fit(self, *args, **kwargs):
        pass

    def predict(self, X):
        return self.model(X[:, 0])


class GaussianProcesses(BaseEstimator, RegressorMixin):
    def __init__(self, kernels, options=None):
        self.kernels = kernels
        self.options = options

    def fit(self, X, y, restarts=None):
        kernels = deepcopy(self.kernels)
        input_dim = {'input_dim': X.shape[1]}
        if self.options is None:
            options = [input_dim] * len(self.kernels)
        elif len(self.kernels) == len(self.options):
            options = deepcopy(self.options)
            for opt in options:
                opt.update(input_dim)
        else:
            raise(ValueError)

        kernel = np.sum([kern(**opt) for kern, opt in zip(kernels, options)])
#         self.output_scaler = StandardScaler()
#         y = self.output_scaler.fit_transform(y)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        self.model = gpy.models.GPRegression(X, y, kernel)
        if restarts is None:
            self.model.optimize(messages=False, max_iters=1000)
        else:
            self.model.optimize(num_restarts=restarts)

    def predict(self, X):
        #         a.inverse_transform
        return self.model.predict(X)[0][:, 0]

    def predict_variance(self, X):
        return self.model.predict(X)[1]

    def predict_gradient(self, X):
        #         print('UNSCALED GRADIENTS')
        return self.model.predictive_gradients(X)[0][:, :, 0]
#     def predict_quantiles()


class GaussianProcessesEDR(GaussianProcesses):

    def __init__(self, edr_ndim, kernels, options=None, mode='once'):
        super(GaussianProcessesEDR, self).__init__(kernels, options)
        self.edr_ndim = edr_ndim
        self.mode = mode
        self.projection_ = None
        self._full_model = None
#         self.explained_variance_ = None

    def fit(self, X, y):
        if self.mode == 'once':
            self.fit_project(X, y, self.edr_ndim)
            self._full_model = self.model
        elif 'iter' in self.mode:
            X_projected = X.copy()
            for dim in range(X.shape[1] - 1, self.edr_ndim - 1, -1):
                self.fit_project(X_projected, y, dim)
                X_projected = self.project(X)
                if dim == (X.shape[1] - 1):
                    self._full_model = self.model
        else:
            raise ValueError('Unknown mode')

        # build the final model
        super(GaussianProcessesEDR, self).fit(self.project(X), y)

    def predict(self, X):
        return super(GaussianProcessesEDR, self).predict(self.project(X))

    def fit_project(self, X, y, ndim):
        super(GaussianProcessesEDR, self).fit(X, y)
        grad = self.predict_gradient(X)
        pca = PCA(n_components=ndim)
        pca.fit(grad)
        self.projection_ = (pca.components_.T if self.projection_ is None else
                            np.dot(self.projection_, pca.components_.T))

    def project(self, X):
        if self.projection_ is None:
            raise ValueError('Fit the model before projection')
        return np.dot(X, self.projection_)
    # TODO: explained variance calculation


class GaussianProcessesEDR_custom(GaussianProcessesEDR):
    def fit_project(self, X, y, ndim):
        super(GaussianProcessesEDR, self).fit(X, y)
        grad = self.predict_gradient(X)
        pca = PCA(n_components=ndim)
        pca.fit(grad[:, 1:])
        comp_ = np.hstack([np.eye(len(pca.components_) + 1, 1),
                           np.vstack([np.zeros([1, pca.components_.shape[1]]),
                                      pca.components_])])
        self.projection_ = (comp_.T if self.projection_ is None else
                            np.dot(self.projection_, comp_.T))
