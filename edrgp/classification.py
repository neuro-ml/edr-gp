from sklearn.base import ClassifierMixin
from GPy.models import GPClassification
from .base import _BaseGP


class GaussianProcessClassifier(_BaseGP, ClassifierMixin):

    def _get_model(self, X, y, kernel):
        return GPClassification(X, y, kernel, self.Y_metadata,
                                self.mean_function)
