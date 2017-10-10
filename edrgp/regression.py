from sklearn.base import RegressorMixin
from GPy.models import GPRegression
from .base import _BaseGP


class GaussianProcessRegressor(_BaseGP, RegressorMixin):

    def __init__(self, kernels=None, kernel_options=None, Y_metadata=None,
                 normalizer=None, noise_var=1.0, mean_function=None):
        self.normalizer = normalizer
        self.noise_var = noise_var

        super(GaussianProcessRegressor, self).__init__(
            kernels, kernel_options, Y_metadata, mean_function)

    def _get_model(self, X, y, kernel):
        return GPRegression(X, y, kernel, self.Y_metadata, self.normalizer,
                            self.noise_var, self.mean_function)
