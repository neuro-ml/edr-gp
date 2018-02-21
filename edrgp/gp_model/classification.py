"""Class for classification based on `GPy` classification"""

from sklearn.base import ClassifierMixin
from GPy.models import GPClassification as _GPClassification
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
