"""Base class for estimators from GPy to use them as sklearn estimators"""

import numpy as np
from sklearn.utils import check_X_y, assert_all_finite, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator
from copy import deepcopy
from GPy import kern as gpy_kern
from abc import ABCMeta, abstractmethod
import six


class _BaseGP(six.with_metaclass(ABCMeta, BaseEstimator)):
	"""Base class for all estimators in EDR-GP

	Parameters
	----------
	kernels : str or list of str, optional
		Kernel for GPy model.
		If string, that kernel should be in GPy.kern.
		If list of str, the sum of kernels is used.
		Default="RBF"
	kernel_options : dict or list of dict
		Kernel options to be set for kernels.
		If `kernels` is str, `kernel_options` should be dict.
		Default={'input_dim': X.shape[1]}
	Y_metadata
		Metadata assosiated with points.
	mean_function :
		???

	Attributes
	----------
	estimator_ : object
		GPy estimator fitted to data.
	n_features_ : int
		Number of features in fitted data.

	"""

    def __init__(self, kernels=None, kernel_options=None, Y_metadata=None,
                 mean_function=None):
        self.kernels = kernels
        self.kernel_options = kernel_options
        self.Y_metadata = Y_metadata
        self.mean_function = mean_function

    def fit(self, X, y, method='optimize', **opt_kws):
    	"""Fit the model according to the given training data

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            Training set.
        y : array-like of shape [n_samples]
            Target values.
        method : {'optimize', 'optimize_restarts'}
        	Invokes passed method to fit GPy model. 
        	For 'optimize_restarts' perform random restarts of the
        	model, and set the model to the best

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """
        X, y = self._check_data(X, y)

        kernel = self._make_kernel(X)
        self.estimator_ = self._get_model(X, y, kernel)
        self.n_features_ = X.shape[1]

        opt_kws.setdefault('messages', False)
        opt_kws.setdefault('max_iters', 1000)
        getattr(self.estimator_, method)(**opt_kws)
        return self

    def _check_data(self, X, y):
    	"""Check data before fitting the model."""
        X, y = check_X_y(X, y, accept_sparse=False)
        if self._estimator_type == 'classifier':
            check_classification_targets(y)
        y = y[:, np.newaxis]
        return X, y

    def _check_input(self, X):
    	"""Check X before predicting
		
		X.shape[1] should be equal `n_features_`.
    	"""
        X = check_array(X, accept_sparse=False)
        if X.shape[1] != self.n_features_:
            raise ValueError("X has {} features per sample; expecting {}"
                             .format(X.shape[1], self.n_features_))
        return X

    def _make_kernel(self, X):
    	"""Create kernel for GPy model."""

        # kernel will be initiated as 'RBF' in model automatically
        if self.kernels is None:
            return self.kernels

        if isinstance(self.kernels, str):
            self.kernels = [self.kernel]

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

    @abstractmethod
    def _get_model(self, X, y, kenrel):
        pass

    def _check_predict(self, X):
    	"""Check that X is correct and model is fitted"""
        X = self._check_input(X)
        check_is_fitted(self, 'estimator_')
        return X

    def predict(self, X):
    	"""Predict the target for the new points

    	Parameters
    	----------
    	X : array-like, shape = [n_samples, `n_features_`]

    	Returns
    	-------
    	y_proba : ndarray
    		Returns the posterior probability of the sample for second
    		class in the model

    	"""
        X = self._check_predict(X)
        y_pred = self.estimator_.predict(X)[0][:, 0]
        assert_all_finite(y_pred)
        return y_pred

    def predict_variance(self, X):
    	"""Predict the target for the new points.

    	Parameters
    	----------
    	X : array-like, shape = [n_samples, `n_features_`]

    	Returns
    	-------
    	y_proba : ndarray
    		Returns the posterior variance of the sample for second
    		class in the model

    	"""
        X = self._check_predict(X)
        return self.estimator_.predict(X)[1]

    def predict_gradient(self, X):
    	"""Compute the derivative of the predicted latent function.

    	Parameters
    	----------
    	X : array-like, shape = [n_samples, `n_features_`]
		
		Returns
		-------
		grads : ndarray, shape = [n_samples, 'n_features_']
			Returns the gradients of the sample.
    	"""
        X = self._check_predict(X)
        return self.estimator_.predictive_gradients(X)[0][:, :, 0]
