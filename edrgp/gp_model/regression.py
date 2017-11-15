"""Classes for regression based on `GPy` regression"""

from sklearn.base import RegressorMixin
from GPy.models import GPRegression
from .base import _BaseGP


class GaussianProcessRegressor(_BaseGP, RegressorMixin):
    """Gaussian Process Regressor based on `GPy` regressor.

    Parameters
    ----------
    kernels : str or list of str, optional
        Kernel for `GPy` model.
        If string, that kernel should be in `GPy.kern`.
        If list of str, the sum of kernels is used.
        Default="RBF"
    kernel_options : dict or list of dict, optional
        Kernel options to be set for kernels.
        If `kernels` is str, `kernel_options` should be dict.
        Default={'input_dim': X.shape[1]}
    Y_metadata : optional
        Metadata assosiated with points.
    normalizer : object, optional
        Normalize the outputs Y. Prediction will be un-normalized using
        this normalizer. If normalizer is None, we will normalize using
        ``GPy.util.normalizer.Standardize``. If normalizer is False, no
        normalization will be done.
    noise_var : float, optional
        The noise variance for Gaussian likelhood, defaults to 1.
    mean_function : optional
    method : {'optimize', 'optimize_restarts'}, optional
        Invokes passed method to fit `GPy` model. 
        For 'optimize_restarts' perform random restarts of the
        model, and set the model to the best.

    Attributes
    ----------
    estimator_ : object
        `GPy` estimator fitted to data.
    n_features_ : int
        Number of features in fitted data.
    """

    def __init__(self, kernels=None, kernel_options=None, Y_metadata=None,
                 normalizer=None, noise_var=1.0, mean_function=None,
                 method='optimize'):
        self.normalizer = normalizer
        self.noise_var = noise_var

        super(GaussianProcessRegressor, self).__init__(
            kernels, kernel_options, Y_metadata, mean_function, method)

    def _get_model(self, X, y, kernel):
        """Returns the `GPy` regression model with initialized params

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training set.
        y : array-like, shape (n_samples)
            Target values.
        kernel : object
            `GPy` kernel.

        Returns
        -------
        model : object
            Returns the `GPy` regression model.

        """
        return GPRegression(X, y, kernel, self.Y_metadata, self.normalizer,
                            self.noise_var, self.mean_function)
