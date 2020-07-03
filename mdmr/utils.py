import numpy as np
from inspect import isclass

def check_symmetric(X):
    """
    check whether input matrix is symmetric or not

    """ 
    if not np.all(X==X.T):
        raise AttributeError("Distance matrix is not symmetric")
    

def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = all_or_any([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [v for v in vars(estimator)
                 if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        raise AttributeError(msg % {'name': type(estimator).__name__})
    