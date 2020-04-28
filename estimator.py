''' BP: Feb 12, 2020
Random features classifier for time series data'''

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
import numpy as np
from scipy import signal


## nonlinearities
def relu(x, thrsh=0):
    return np.maximum(x, thrsh)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RFClassifier(BaseEstimator, ClassifierMixin):
    """
    Random features with SVM classification
    """
    
    def __init__(self, width=500, weights='white noise', nonlinearity=relu, clf=None, weight_fun=None, clf_args={}, 
                 seed=None):
        self.width = width
        self.weights = weights
        self.nonlinearity = nonlinearity
        self.clf = clf
        self.weight_fun = weight_fun
        self.clf_args = clf_args
        self.seed = seed

    def fit(self, X, y):
        # check params
        if X.shape[0] != y.shape[0]:
            raise ValueError('dimension mismatch')
        n = X.shape[1]
        
        if self.seed is None:
            self.seed = np.random.randint(1000)
    
        if self.clf is None:
            self.clf = LinearSVC(random_state=self.seed, tol=1e-4, max_iter=1000)
        else:
            self.clf = self.clf(**self.clf_args)
        
        if self.weight_fun is not None:
            self.W_ = self.weight_fun(self.width, n)
        else:
            self.W_ = \
            random_feature_matrix(self.width, n, self.weights, self.seed)
        H = self.nonlinearity(X @ self.W_)
        
        #fit classifier
        self.clf.fit(H, y)
        self._fitted = True
        return self

    def transform(self, X):
        H = self.nonlinearity(X @ self.W_)
        return H
    
    def predict(self, X):
        H = self.nonlinearity(X @ self.W_)
        return self.clf.predict(H)
    
    def score(self, X, y):
        H = self.nonlinearity(X @ self.W_)
        return self.clf.score(H, y)

def gaussian(X, mu, sigma):
    return np.exp(-np.abs(mu - X) ** 2/ (2 * sigma **2))


def random_feature_matrix(M, N, weights='white noise', rand_seed=None):
    ''' 
    Generate a size (M, N) random matrix from the specified distribution
    
    Parameters
    ----------
    
    M: number of rows
    
    N: number of columns
    
    weights: string or function, default 'gaussian.'
    If 'unimodal', entries are gaussians with means ~ Unif(0, 1), and std. dev ~ Unif(0.1, N).
    If 'white noise', entries are drawn ~ N(0, 1).
    Or can be a function handle taking arguments (M, N)
    '''
    if rand_seed is None:
        rand_seed = np.random.randint(1000)
    np.random.seed(rand_seed)

    if weights == 'unimodal':
        mu = np.random.uniform(0, N, (M, 1))
        sigma = np.random.uniform(0.1, N/4, (M, 1))
        k = np.arange(0, N)
        J = np.array([gaussian(k, m, s) for (m, s) in zip(mu, sigma)]) * np.random.randint(1, 20, (M, 1))

    elif weights == 'white noise':
        J = np.random.randn(M, N)
    
    elif weights == 'identity':
        J = np.eye(N, N)

    else:
        J = weights(M, N)
    return J.T

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return signal.lfilter(b, a, data)

def bp_weights(M, N, lowcut, highcut, fs):
    J = np.random.randn(M, N)
    J = butter_bandpass_filter(J, lowcut, highcut, fs)
    return J.T

def bp_weights_gaus(M, N, lowcut, highcut, fs):
    W = np.zeros((M, N))

    t_points = np.arange(N) / fs
    wk = 2 * np.pi * np.arange(lowcut, highcut + 1)
    
    c = 1 / len(wk)
    Sk = 1
    for i in range(M):
        Ak = np.random.normal(size=(len(wk), 2))
        for j, t in enumerate(t_points):
            W[i, j] = c / np.sqrt(np.pi) * np.sum(Sk * (Ak[:, 0] * np.cos(wk * t) + Ak[:, 1] * np.cos(wk * t)))
    return W.T




