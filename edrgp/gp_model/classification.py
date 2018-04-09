"""Class for classification based on `GPy` classification"""

from sklearn.base import ClassifierMixin
from GPy.models import GPClassification as _GPClassification
from GPy.models import SparseGPClassification as _SGPClassification
from .base import _BaseGP


class GaussianProcessClassifier(_BaseGP, ClassifierMixin):
    """GaussianProcessClassifier based on `GPy` classifier

    Parameters
    ----------
    kernels : str or list of str, optional (default="RBF")
        Kernel for `GPy` model.
        If string, that kernel should be in `GPy.kern`.
        If list of str, the sum of kernels is used.
    kernel_options : dict or list of dict, optional
        Kernel options to be set for kernels.
        If `kernels` is str, `kernel_options` should be dict.
        Default={'input_dim': X.shape[1]}
    Y_metadata : optional
        Metadata assosiated with points.
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
                 mean_function=None, method='optimize'):
        self.kernels = kernels
        self.kernel_options = kernel_options
        self.Y_metadata = Y_metadata
        self.mean_function = mean_function
        self.method = method

    def _get_model(self, X, y, kernel):
        """Returns the `GPy` classification model with initialized params

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
            Returns the `GPy` classification model.
        """
        return _GPClassification(X, y, kernel, self.Y_metadata,
                                 self.mean_function)

    def predict(self, X):
        """Predict the target for the new points

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Testing points.

        Returns
        -------
        y : ndarray
            Returns 1 if the posterior probability of second class is higher
            than 0.5, else returns 0.

        """
        return super(GaussianProcessClassifier, self).predict(X) > 0.5

    def predict_proba(self, X):
        """Predict the target for the new points

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Testing points.

        Returns
        -------
        y_proba : ndarray
            Returns the posterior probability of the
            sample for second class in the model.

        """
        return super(GaussianProcessClassifier, self).predict(X)


class SparseGaussianProcessClassifier(_BaseGP, ClassifierMixin):
    """SparseGaussianProcessClassifier based on `GPy` sparse classifier

    Parameters
    ----------
    kernels : str or list of str, optional (default="RBF")
        Kernel for `GPy` model.
        If string, that kernel should be in `GPy.kern`.
        If list of str, the sum of kernels is used.
    kernel_options : dict or list of dict, optional
        Kernel options to be set for kernels.
        If `kernels` is str, `kernel_options` should be dict.
        Default={'input_dim': X.shape[1]}
    likelihood : object
        GPy likelihood, defaults to Binomial with probit link_function
    Z : array-like, shape (num_inducing, input_dim), optional
        Inducing inputs
    num_inducing : int, optional (default=10)
        Number of inducing points (Ignored if Z is not None)
        If specified select inducing points randomly. 
    Y_metadata : optional
        Metadata assosiated with points.
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

    def __init__(self, kernels=None, kernel_options=None, likelihood=None,
                 Z=None, num_inducing=10, Y_metadata=None, method='optimize'):
        self.kernels = kernels
        self.kernel_options = kernel_options
        self.likelihood = likelihood
        self.Z = Z
        self.num_inducing = num_inducing
        self.Y_metadata = Y_metadata
        self.method = method

    def _get_model(self, X, y, kernel):
        _SGPClassification(X, y, likelihood=self.likelihood, kernel=kernel,
                           Z=self.Z, num_inducing=self.num_inducing,
                           Y_metadata=self.Y_metadata)

    def predict(self, X):
        """Predict the target for the new points

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Testing points.

        Returns
        -------
        y : ndarray
            Returns 1 if the posterior probability of second class is higher
            than 0.5, else returns 0.

        """
        return super(SparseGaussianProcessClassifier, self).predict(X) > 0.5

    def predict_proba(self, X):
        """Predict the target for the new points

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Testing points.

        Returns
        -------
        y_proba : ndarray
            Returns the posterior probability of the
            sample for second class in the model.

        """
        return super(SparseGaussianProcessClassifier, self).predict(X)
