"""Classes for regression based on `GPy` regression"""

from sklearn.base import RegressorMixin
from GPy.models import GPRegression as _GPRegression
from GPy.models import SparseGPRegression as _SparseGPRegression
from GPy.models import GPHeteroscedasticRegression as _GPHeteroscedasticRegression
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
                 normalizer=True, noise_var=1.0, mean_function=None,
                 method='optimize'):
        self.normalizer = normalizer
        self.noise_var = noise_var
        self.kernels = kernels
        self.kernel_options = kernel_options
        self.Y_metadata = Y_metadata
        self.mean_function = mean_function
        self.method = method

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
        return _GPRegression(X, y, kernel, self.Y_metadata, self.normalizer,
                             self.noise_var, self.mean_function)


class SparseGaussianProcessRegressor(_BaseGP, RegressorMixin):
    """Sparse Gaussian Process Regressor based on `GPy` sparse regressor.

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
    Z : array-like, shape (num_inducing, input_dim), optional
        Inducing inputs
    num_inducing : int, optional (default=10)
        Number of inducing points (Ignored if Z is not None)
        If specified select inducing points randomly. 
    Y_metadata : optional
        Metadata assosiated with points.
    X_variance : array-like, optional
        Input uncertainties, one per input X
    normalizer : object, optional
        Normalize the outputs Y. Prediction will be un-normalized using
        this normalizer. If normalizer is None, we will normalize using
        ``GPy.util.normalizer.Standardize``. If normalizer is False, no
        normalization will be done.
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

    def __init__(self, kernels=None, kernel_options=None, Z=None,
                 num_inducing=10, Y_metadata=None, X_variance=None,
                 normalizer=True, mean_function=None, method='optimize'):
        self.kernels = kernels
        self.kernel_options = kernel_options
        self.Z = Z
        self.num_inducing = num_inducing
        self.Y_metadata = Y_metadata
        self.X_variance = X_variance
        self.normalizer = normalizer
        self.mean_function = mean_function
        self.method = method

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
        return _SparseGPRegression(X, y, kernel=kernel, Z=self.Z,
                                   num_inducing=self.num_inducing,
                                   X_variance=self.X_variance,
                                   mean_function=self.mean_function,
                                   normalizer=self.normalizer)

# class GaussianProcessHeteroscedasticRegressor(_BaseGP, RegressorMixin):
#     """Heteroscedastic Gaussian Process Regressor based on `GPy` regressor.

#     Parameters
#     ----------
#     kernels : str or list of str, optional
#         Kernel for `GPy` model.
#         If string, that kernel should be in `GPy.kern`.
#         If list of str, the sum of kernels is used.
#         Default="RBF"
#     kernel_options : dict or list of dict, optional
#         Kernel options to be set for kernels.
#         If `kernels` is str, `kernel_options` should be dict.
#         Default={'input_dim': X.shape[1]}
#     Y_metadata : optional
#         Metadata assosiated with points.
#     method : {'optimize', 'optimize_restarts'}, optional
#         Invokes passed method to fit `GPy` model. 
#         For 'optimize_restarts' perform random restarts of the
#         model, and set the model to the best.

#     Attributes
#     ----------
#     estimator_ : object
#         `GPy` estimator fitted to data.
#     n_features_ : int
#         Number of features in fitted data.
#     """
#     def __init__(self, kernels=None, kernel_options=None, Y_metadata=None,
#                  method='optimize'):
#         self.kernels = kernels
#         self.kernel_options = kernel_options
#         self.Y_metadata = Y_metadata
#         self.method = method

#     def _get_model(self, X, y, kernel):
#         """Returns the `GPy` regression model with initialized params

#         Parameters
#         ----------
#         X : array-like, shape (n_samples, n_features)
#             Training set.
#         y : array-like, shape (n_samples)
#             Target values.
#         kernel : object
#             `GPy` kernel.

#         Returns
#         -------
#         model : object
#             Returns the `GPy` regression model.
#         """
#         return _GPHeteroscedasticRegression(X, y, kernel, 
#                                             Y_metadata=self.Y_metadata)
