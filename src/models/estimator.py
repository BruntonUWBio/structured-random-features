from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import scipy.linalg
from scipy.spatial.distance import pdist, squareform
import numpy as np
import numpy.linalg as la

class RFClassifier(BaseEstimator, ClassifierMixin):
    """
    Random feature classifier.

    This class projects inputs onto randomly generated weights and 
    classifies them using a linear classifier.

    Parameters
    ----------

    width : int
        Number of random weights the input is 
        projected onto.

    nonlinearity: callable
        Specifies the non-linear transformation that is 
        applied to the randomly transformed inputs. The call signature
        is ``fun(x)``.
    
    weight_fun: callable
        Specifies the function to generate the random weights. The call signature is 
        ``fun(width, n_features, **kwargs, random_state). ``
        Here, `width` is a positive int and specifies the 
        number of random weights. Here, `n_features` is a positive 
        int and specifies the size of each random weight. `fun`
        must return array_like with shape (width, n_features) i.e.
        each row corresponds to a random weight.

    kwargs: dict, optional
        Additional arguments to be passed to the weight
        function. If for example if the weight function has the 
        signature ```fun(width, n_features, a, b, c, random_state, )```, 
        then `kwargs` must be a dict with three parameters and their
        keywords. 
    
    bias: ndarray of shape (width,) or (1,)
        The bias term for the randomly transformed input. If (1,), same
        bias is used for all the random weights.

    clf : sklearn linear classifier object, eg: logistic regression, 
        linear SVM. Specifies the linear classifier used for 
        classification.

    random_state: int, default=None
        Used to set the seed when generating random weights.
    
    Attributes
    -------

    W_ : ndarray of shape (width, n_features)
        Random weights that are generated.

    b_ : ndarray of shape (width, ) or (1,)
        The bias term for the randomly transformed input.

    H_ : {array-like} of shape (n_samples, width)
        Transformed train input, where n_samples is the number of samples
        and width is the number of features.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function of the 
        classifier.
    
    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function of the
        classifier.

    Examples
    --------

    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from estimator import RFClassifier, V1_inspired_weights

    X, y = load_digits(return_X_y=True)
    logit = LogisticRegression(solver='saga')
    relu = lambda x: np.maximum(0, x)
    kwargs = {'t': 5, 'l': 3}
    clf = RFClassifier(width=20, weight_fun=V1_inspired_weights, 
    kwargs=kwargs, bias=2, nonlinearity=relu, clf=logit, random_state=22)
    clf.fit(X, y)
    clf.score(X, y)
    
    """

    def __init__(self, width, weight_fun, bias, nonlinearity, 
    clf, kwargs=None, random_state=None):
        self.width = width
        self.nonlinearity = nonlinearity
        self.weight_fun = weight_fun
        self.bias = bias
        self.clf = clf
        self.kwargs = kwargs
        self.random_state = random_state
    
    def _fit_transform(self, X):
        """
        Project the train input onto the random weights.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of sampless
            and n_features is the number of features.

        Returns
        -------
        self 
        
        """
        n_features = X.shape[1]
        if self.kwargs is not None:
            self.W_ = self.weight_fun(self.width, n_features, 
                                        **self.kwargs, random_state=self.random_state)
        else:
            self.W_ = self.weight_fun(self.width, n_features, 
                            random_state=self.random_state)

        self.b_ = self.bias
        self.H_ = self.nonlinearity(np.dot(X, self.W_.T) + self.b_)

    def fit(self, X, y):
        """
        Fit the model according to the given training data. 

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator
        """
        self._fit_transform(X)
        self.clf.fit(self.H_, y)
        self.coef_ = self.clf.coef_
        self.intercept_ = self.clf.intercept_
        return self

    def transform(self, X):
        """
        Project test input onto the random weights.

        Parameters
        ----------

        X : {array-like} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of sampless
            and n_features is the number of features.

        Returns
        -------

        H : {array-like} of shape (n_samples, width)
            Transformed input, where n_samples is the number of samples
            and n_features is the number of features.
        """
        check_is_fitted(self, ["W_", "b_"])
        H = self.nonlinearity(np.dot(X, self.W_.T) + self.b_)
        return H

    def score(self, X, y):
        """
        Returns the score on the given test data and labels.

        Parameters
        ---------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        ------
        score : float
        """
        H = self.transform(X)
        check_is_fitted(self, ["coef_", "intercept_"])
        score = self.clf.score(H, y)
        return score
        

# def classical_weights(M, N, random_state=None):
#     """"
#     Generates classical random weights. W ~ N(0, 1).

#     Parameters
#     ----------

#     M : int
#         Number of random weights

#     N : int
#         Number of features
    
#     random_state : int, default=None
#         Used to set the seed when generating random weights.

#     Returns
#     -------

#     W : array-like of shape (M, N)
#         Random weights.
#     """
#     np.random.seed(random_state)
#     dft = scipy.linalg.dft(N, scale=None)
#     rand = np.random.normal(0, 1, size=(M, N, 2)).view(np.complex).squeeze(axis=2)
#     W = np.dot(rand, dft).real
#     W /= np.std(W, axis=1).reshape(-1, 1) # original
# #     W /= la.norm(W, axis=1).reshape(-1, 1)
#     W /= la.norm(W, axis=0).reshape(1, -1)
#     return W

# def haltere_inspired_weights(M, N, lowcut, highcut, random_state=None):
#     """
#     Generates random weights with tuning similar to mechanosensory 
#     neurons in insect halteres. Weights are band-limited gaussian 
#     process with fourier eigenbasis.

#     Parameters
#     ----------

#     M : int
#         Number of random weights

#     N : int
#         Number of features

#     lowcut: int, column of the (N x N) DFT matrix
#         Low end of the frequency band. 

#     highcut: int, column of the (N x N) DFT matrix
#         High end of the frequency band.
    
#     random_state : int, default=None
#         Used to set the seed when generating random weights.
    
#     Returns
#     -------

#     W : array-like of shape (M, N)
#         Random weights.
#     """
#     np.random.seed(random_state)
#     dft = scipy.linalg.dft(N, scale=None)
#     rand = np.zeros((M, N), dtype=complex)
#     rand[:, lowcut:highcut] = np.random.normal(0, 1, 
#                 size=(M, highcut-lowcut, 2)).view(np.complex).squeeze()
#     W = np.dot(rand, dft).real
#     W /= la.norm(W, axis=0).reshape(1, -1)
#     W /= np.std(W, axis=1).reshape(-1, 1) # original
# #     W /= la.norm(W, axis=1).reshape(-1, 1)
#     return W


def V1_inspired_kernel_matrix(N, s, f, center, scale=1):
    """
    Generates the kernel matrix for Gaussian Process with non-stationary 
    covariance. This kernel matrix will be used to generate random 
    features inspired from the receptive-fields of V1 neurons.

    K(x, y) = exp(|x - y|^2/2t) * exp(|x - m|^2/2l) * exp(|y - m|^2/2l)

    Parameters
    ----------

    N : int or tuple of length 2
        Dimensions of random features.
        int for square features of size sqrt(N) x sqrt(N)
        tuple for features of size N[0] x N[1]

    s : float
        Determines the size of the random weights 

    f : float
        Determines the spatial frequency of the random weights  
    
    center : tuple (2, 1)
        Determines the center of the random weights.

    Returns
    -------

    K : array-like of shape (N, N) or (N[1] * N[2], N[1] * N[2])
        Kernel matrix
    """
    
    from scipy.spatial.distance import pdist, squareform
    
    # for random features of size sqrt(N) x sqrt(N)
    if type(N) is int:
        x = np.arange(np.sqrt(N))
        yy, xx = np.meshgrid(x, x)
        dim = N

    # for random features of size N[0] x N[1]
    elif type(N) is tuple:
        x = np.arange(N[0])
        y = np.arange(N[1])
        yy, xx = np.meshgrid(y, x)
        dim = N[0] * N[1]

    grid = np.column_stack((xx.flatten(), yy.flatten()))

    a = squareform(pdist(grid, 'sqeuclidean'))
    b = la.norm(grid - center, axis=1) ** 2
    c = b.reshape(-1, 1)
    K = np.exp(-a / (2 * f ** 2)) * np.exp(-b / (2 * s ** 2)) * np.exp(-c / (2 * s ** 2))
    K += 1e-5 * np.eye(dim)
    K *= (scale * dim / np.trace(K))
    return K


def V1_inspired_weights_for_center(N, s, f, center, scale=1, random_state=None):
    """
    Generates a random weight for one given center by sampling a 
    non-stationary Gaussian Process.

    Parameters
    ----------

    N : int
        Number of features 

    s : float
        Determines the size of the random weights

    f : float
        Determines the spatial frequency of the random weights 
    
    center : tuple (2, 1)
        Determines the center of the random weights.

     random_state : int, default=None
        Used to set the seed when generating random weights.

    Returns
    -------

    W : (array-like) of shape (N,)
        A random weight
    """
    np.random.seed(random_state)
    K = V1_inspired_kernel_matrix(N, s, f, center, scale)
    L = la.cholesky(K)
    
    if type(N) is int:
        dim = N
    
    if type(N) is tuple:
        dim = N[0] * N[1]
    
    W = np.dot(L, np.random.randn(dim))
    return W

def V1_inspired_weights(M, N, s, f, scale=1, random_state=None):
    """
    Generate random weights inspired by the tuning properties of the 
    neurons in Primary Visual Cortex (V1).

    Parameters
    ----------

    M : int
        Number of random weights

    N : int
        Number of features
    
    s : float
        Determines the size of the random weights

    f : float
        Determines the spatial frequency of the random weights 

    random_state : int, default=None
        Used to set the seed when generating random weights.

    Returns
    -------

    W : array-like of shape (M, N)
        Random weights.

    """
    np.random.seed(random_state)
    
       
    if type(N) is int:
        centers = np.random.randint(int(np.sqrt(N)), size=(M, 2))
    
    if type(N) is tuple:
        centers = np.random.randint((N[0], N[1]), size=(M, 2))

    # centers = np.random.randint(int(np.sqrt(N)), size=(M, 2))

    W = np.empty(shape=(0, N))
    for center in centers:
        w = V1_inspired_weights_for_center(N, s, f, center, scale,
                                            random_state=random_state)
        W = np.row_stack((W, w.T))
    return W
    

def relu(x, thrsh=0):
    """
    Rectified Linear Unit

    Parameters
    ----------

    x : {array-like} or int
        Input data
    
    thrsh: int
        threshold for firing

    Returns
    -------

    y : {array-like} or int
        Output
    """
    return np.maximum(x, thrsh)

def poly(x, power=2):
    """
    Polynomial function. Raises input to specified power.

    Parameters
    ----------

    x : {array-like} or int
        Input
    
    power: int
        Degree of the polynomial

    Returns
    -------

    y : {array-like} or int
        Output
    """
    return np.power(x, power)



def clf_wrapper(RFClassifier, params, X_train, y_train, X_test, y_test, return_clf=False):
    """
    Wrapper function for RFClassifier. Fits on train data and evaluates on the test data.
    
    Parameters
    ----------
    
    RFClassifier : class
        Random feature classifier class
    
    params : dict
        Parameters for the RFClassifier
    
    X_train : array-like of shape (n_samples, n_features)
        Training data
    
    y_train : array-like of shape (n_samples,)
        Training labels
        
    X_test : array-like of shape (n_samples, n_features)
        Test data
       
    y_test : array-like of shape (n_samples,)
        Test labels
    
    return_clf : bool, default=False
        Returns the fitted RFClassifier
   
   Returns
   -------

   train_error : float
       training error
   
   test_error : float
       test error
   
   clf : class
       Fitted RFCLassifier class
  
    """
    
    clf = RFClassifier(**params)
    clf.fit(X_train, y_train)
    test_error = 1 - clf.score(X_test, y_test)
    train_error = 1 - clf.score(X_train, y_train)
    
    if return_clf is True:
        return train_error, test_error, clf
    else:
        return train_error, test_error
    
    
def parallelized_clf(RFClassifier, params, X_train, y_train, X_test, y_test, n_iters=5, return_clf=False):
    """
    Runs random feature classifier multiple times using the Dask framework.
    
    Parameters
    ----------
    
    RFClassifier : class
        Random feature classifier class
        
    params : dict
        Parameters for the RFClassifier
    
    X_train : array-like of shape (n_samples, n_features)
        Training data
    
    y_train : array-like of shape (n_samples,)
        Training labels
        
    X_test : array-like of shape (n_samples, n_features)
        Test data
       
    y_test : array-like of shape (n_samples,)
        Test labels
        
    n_iters : int
        Number of times to run the RFClassifer on train data
        
    return_clf: bool
        returns the weights of the classifier if 'True'
   
   
   Returns
   -------

   mean_train_error : float
       average training error for all of the n_iter classifers
   
   std_train_error : float
       standard deviation of the training error for all of the n_iter classifiers
       
   mean_test_error : float
       average test error for all of the n_iter classifiers
       
   std_test_error : float
       standard deviation of the test error for all the n_iter classifiers
    """
    
    import dask

    lazy_results = []
    for i in range(n_iters):
        lazy_clf = dask.delayed(clf_wrapper)(RFClassifier, params, X_train, y_train, X_test, y_test, return_clf=return_clf)
        lazy_results.append(lazy_clf)
    futures = np.array(dask.compute(*lazy_results))

    train_error = futures[:, 0]
    test_error = futures[:, 1]
    mean_train_error = np.mean(train_error)
    std_train_error = np.std(train_error)
    mean_test_error = np.mean(test_error)
    std_test_error = np.std(test_error)
    
    if return_clf is True:
        clf = futures[:, 2]
        return mean_train_error, std_train_error, mean_test_error, std_test_error, clf
    else:
        return mean_train_error, std_train_error, mean_test_error, std_test_error
    

def classical_covariance_matrix(N, scale=1):
    """
    Generates the (N x N) covariance matrix for Gaussian Process with identity covariance. 
    This kernel matrix will be used to generate random weights that are traditionally used 
    in kernel methods.

    K(x, y) = \delta_{xy}

    Parameters
    ----------

    N : int
        Number of features 

    scale: float, default=1
        Used to adjust the magnitude of trace of the cov. matrix

    Returns
    -------

    C : array-like of shape (N, N)
        Covariance matrix
    """
    C = np.eye(N) * scale
    return C

def classical_weights(M, N, scale=1, random_state=None):
    """"
    Generates classical random weights. W ~ N(0, 1).

    Parameters
    ----------

    M : int
        Number of random weights

    N : int
        Number of features

    scale : float, default=1
        Used to adjust the magnitude of the cov. matrix that generates the weights
    
    random_state : int, default=None
        Used to set the seed when generating random weights.

    Returns
    -------

    W : array-like of shape (M, N)
        Random weights.
    """
    np.random.seed(random_state)
    C = classical_covariance_matrix(N, scale)
    L = la.cholesky(C)
    W = np.dot(L, np.random.randn(N, M)).T
    return W


def haltere_covariance_matrix(N, lowcut, highcut, scale=1):
    '''
    Generates the (N x N) covariance matrix for Gaussain Process inspired by the STAs 
    of mechanosensory neurons in insect halteres.
    
    $$k(t, t') = \mathbb{E}[w(t)^T w(t')] =  \sum_{j=0}^{N-1} \lambda_j \cos{\dfrac{i 2 \pi j (t-t')}{N}} $$
    $$ \lambda_j = \begin{cases} 1 & a \leq b \\ 0 & otherwise \end{cases}$$

    Parameters
    ----------

    N: int
        Number of features
    
    lowcut: int
        low end of the frequency band filter

    highcut : int
        high end of the frequency band filter

    Returns
    -------
    C : array-like of shape (N, N) 
        Covariance matrix
    '''
    lamda = np.zeros(N)
    lamda[lowcut:highcut] = 1
    
    grid = np.arange(0, N)
    yy, xx = np.meshgrid(grid, grid)
    diff = xx - yy
    
    C = np.zeros((N, N))
    for j in range(N):
        C += lamda[j] * np.cos(2 * np.pi * j * diff / N)
    C += 1e-5 * np.eye(N)
    diag = np.diag(np.sqrt(scale/np.diag(C)))
    C = diag @ C @ diag
    return C

def haltere_inspired_weights(M, N, lowcut, highcut, scale=1, random_state=None):
    """
    Generates random weights with tuning similar to mechanosensory 
    neurons in insect halteres. Weights are band-limited gaussian 
    process with fourier eigenbasis.

    Parameters
    ----------

    M : int
        Number of random weights

    N : int
        Number of features

    lowcut: int
        Low end of the frequency band. 

    highcut: int
        High end of the frequency band.
    
    random_state : int, default=None
        Used to set the seed when generating random weights.
    
    Returns
    -------

    W : array-like of shape (M, N)
        Random weights.
    """
    np.random.seed(random_state)
    C = haltere_covariance_matrix(N, lowcut, highcut, scale)
    L = la.cholesky(C)
    W = np.dot(L, np.random.randn(N, M)).T
    return W

    
def V1_weights_multiple_scales(M, N, scale=1, random_state=None):
    """
    Generate random weights inspired by the tuning properties of the 
    neurons in Primary Visual Cortex (V1). The spatial frequency are of 
    different scales.

    Parameters
    ----------

    M : int
        Number of random weights

    N : int
        Number of features

    random_state : int, default=None
        Used to set the seed when generating random weights.

    Returns
    -------

    W : array-like of shape (M, N)
        Random weights.

    """
    np.random.seed(random_state)
    centers = np.random.randint(int(np.sqrt(N)), size=(M, 2))
#     params = np.random.uniform(2, 8, (M, 2))
    params = np.random.choice([2, 5, 8], M)

    W = np.empty(shape=(0, N))
    for i, m in enumerate(centers):
        w = V1_inspired_weights_for_center(N, params[i], params[i], m, scale,
                                            random_state=random_state)
        W = np.row_stack((W, w.T))
    return W


def V1_inspired_weights_same_center(M, N, s, f, center, scale=1, random_state=None):
    """
    Generates M random weights for one given center by sampling a 
    non-stationary Gaussian Process.

    Parameters
    ----------

    M: int
        Number of random weights

    N : int
        Number of features 

    s : float
        Determines the size of the random weights

    f : float
        Determines the spatial frequency of the random weights 
    
    center : tuple (2, 1)
        Determines the center of the random weights.

     random_state : int, default=None
        Used to set the seed when generating random weights.

    Returns
    -------

    W : (array-like) of shape (M, N)
        Random weights
    """
    np.random.seed(random_state)
    if type(N) is int:
        dim = N
    
    if type(N) is tuple:
        dim = N[0] * N[1]
        
    
    K = V1_inspired_kernel_matrix(N, s, f, center, scale)
    L = la.cholesky(K)
    W = np.dot(L, np.random.randn(dim, M)).T
    return W    

def haltere_covariance_matrix_decay(N, lowcut, highcut, decay_coef, scale=1):
    '''
    Generates the (N x N) covariance matrix for Gaussain Process inspired by the STAs 
    of mechanosensory neurons in insect halteres. Decaying sinusoids
    
    $$k(t, t') = \mathbb{E}[w(t)^T w(t')] =  \sum_{j=0}^{N-1} \lambda_j \cos{\dfrac{i 2 \pi j (t-t')}{N}} * exp((- \|t- N - 1\| + \|t'- N - 1\|) / a ** 2) $$
    $$ \lambda_j = \begin{cases} 1 & lowcut \leq highcut \\ 0 & otherwise \end{cases}$$

    Parameters
    ----------

    N: int
        Number of features
    
    lowcut: int
        low end of the frequency band filter

    highcut : int
        high end of the frequency band filter
        
    decay_coef : float
        controls the how fast the random features decay
    
    scale: float
        Normalization factor for Tr norm of cov matrix
    
    Returns
    -------
    C : array-like of shape (N, N) 
        Covariance matrix
    '''
    
    lamda = np.zeros(N)
    lamda[lowcut:highcut] = 1

    grid = np.arange(0, N)
    yy, xx = np.meshgrid(grid, grid)
    diff = xx - yy

    # sinusoidal part
    C_cos = np.zeros((N, N))
    for j in range(lowcut, highcut):
        C_cos += lamda[j] * np.cos(2 * np.pi * j * diff / N)

    # exponential part
    C_exp = np.exp(((xx - N) + (yy - N)) / decay_coef ** 2)

    # final covariance matrix
    C = C_cos * C_exp 
    C *= (scale * N / np.trace(C))
    C += 1e-5 * np.eye(N)
    return C

def decaying_haltere_inspired_weights(M, N, lowcut, highcut, decay_coef, scale=1, random_state=None):
    """
    Generates random weights with tuning similar to mechanosensory 
    neurons in insect halteres.

    Parameters
    ----------

    M : int
        Number of random weights

    N : int
        Number of features

    lowcut: int
        Low end of the frequency band. 

    highcut: int
        High end of the frequency band.
        
    decay_coef : float
        controls the how fast the random features decay
    
    random_state : int, default=None
        Used to set the seed when generating random weights.
    
    Returns
    -------

    W : array-like of shape (M, N)
        Random weights.
    """
    np.random.seed(random_state)
    C = haltere_covariance_matrix_decay(N, lowcut, highcut, decay_coef, scale)
    L = la.cholesky(C)
    W = np.dot(L, np.random.randn(N, M)).T
    return W