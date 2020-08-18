''' BP: Feb 12, 2020
Random features classifier for time series data

May 4th 2020(BP): Normalized H by its frobenius norm. 
'''



from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
import numpy as np
import numpy.linalg as la
from scipy import signal
import scipy.linalg
from scipy.spatial.distance import pdist, squareform


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
#         diag = np.diag(1 / (la.norm(H, axis=1) + 1E-3)) # to avoid division by 0
#         H = np.dot(diag, H)
        
        #fit classifier
        self.clf.fit(H, y)
        self._fitted = True
        return self

    def transform(self, X):
        H = self.nonlinearity(X @ self.W_)
#         diag = np.diag(1 / (la.norm(H, axis=1) + 1E-3))
#         H = np.dot(diag, H)
        return H
    
    def predict(self, X):
        H = self.nonlinearity(X @ self.W_)
#         diag = np.diag(1 / (la.norm(H, axis=1) + 1E-3))
#         H = np.dot(diag, H)
        return self.clf.predict(H)
    
    def score(self, X, y):
        H = self.nonlinearity(X @ self.W_)
#         diag = np.diag(1 / (la.norm(H, axis=1) + 1E-3))
#         H = np.dot(diag, H)
        return self.clf.score(H, y)

def gaussian(X, mu, sigma):
    return np.exp(-np.abs(mu - X) ** 2/ (2 * sigma **2))


def random_feature_matrix(M, N, weights='white noise', rand_seed=None):
    ''' 
    Generate a size (M, N) random matrix from the specified distribution
    
    Parameters
    ----------
    
    M: number of samples
    
    N: number of features
    
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
        dft = scipy.linalg.dft(N, scale=None)
        rand = np.random.normal(0, 1, size=(M, N, 2)).view(np.complex).squeeze(axis=2)
        J = (rand @ dft).real
        J /= np.std(J, axis=1).reshape(-1, 1)
    
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

def bp_weights_dft_v1(M, N, lowcut, highcut):
    dft = scipy.linalg.dft(N, scale=None)
    rand = np.zeros((M, N), dtype=complex)
    phi = np.random.uniform(-np.pi, np.pi, (M, highcut - lowcut))
    rand[:, lowcut:highcut] = np.random.normal(size=(M, highcut-lowcut)).astype(complex) * np.e ** (1j * phi)
    W = (rand @ dft).real
    W /= np.std(W, axis=1).reshape(-1, 1)
    return W.T

def bp_weights_dft(M, N, lowcut, highcut):
    dft = scipy.linalg.dft(N, scale=None)
    rand = np.zeros((M, N), dtype=complex)
    rand[:, lowcut:highcut] = np.random.normal(0, 1, size=(M, highcut-lowcut, 2)).view(np.complex).squeeze()
    W = (rand @ dft).real
    W /= np.std(W, axis=1).reshape(-1, 1)
    return W.T

def gabor_kernel_matrix(N, t, l, m):
    x = np.arange(np.sqrt(N))
    yy, xx = np.meshgrid(x, x)
    grid = np.column_stack((xx.flatten(), yy.flatten()))
    
    a = squareform(pdist(grid, 'sqeuclidean'))
    b = la.norm(grid - m, axis=1) ** 2
    c = b.reshape(-1, 1)
    K = 10 * np.exp(-a / (2 * l ** 2)) * np.exp(-b / (2 * t ** 2)) * np.exp(-c / (2 * t ** 2))
    K += 1e-5 * np.eye(N)
    return K

def gabor_random_features_for_center(N, t, l, m, seed=None):
    np.random.seed(seed)
    K = gabor_kernel_matrix(N, t, l, m)
    L = np.linalg.cholesky(K)
    w = np.dot(L, np.random.randn(N))
    return w

def gabor_random_features(M, N, t, l, seed=None):
    np.random.seed(seed)
    centers = np.random.randint(int(np.sqrt(N)), size=(M, 2))

    W = np.empty(shape=(0, N))
    for m in centers:
        w = gabor_random_features_for_center(N, t, l, m, seed=seed)
        W = np.row_stack((W, w.T))
    return W
