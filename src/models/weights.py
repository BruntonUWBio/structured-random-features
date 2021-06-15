import numpy as np
import numpy.linalg as la
from scipy.spatial.distance import pdist, squareform


def sensilla_covariance_matrix(dim, lowcut, highcut, decay_coef=np.inf, scale=1):
    '''
    Generates the (dim x dim) covariance matrix for Gaussain Process inspired by the STAs 
    of mechanosensory neurons in insect halteres. Decaying sinusoids.
    
    $$k(t, t') = \mathbb{E}[w(t)^T w(t')] =  \sum_{j=0}^{dim-1} \lambda_j \cos{\dfrac{i 2 \pi j (t-t')}{dim}} * exp((- \|t- N - 1\| + \|t'- N - 1\|) / decay_coef ** 2) $$
    $$ \lambda_j = \begin{cases} 1 & lowcut \leq highcut \\ 0 & otherwise \end{cases}$$

    Parameters
    ----------

    dim: int
        dimension of each random weight
    
    lowcut: int
        low end of the frequency band filter

    highcut : int
        high end of the frequency band filter
        
    decay_coef : float, default=np.inf
        controls the how fast the random features decay
        With default value, the weights do not decay
    
    scale: float
        Normalization factor for Tr norm of cov matrix
    
    Returns
    -------
    C : array-like of shape (dim, dim) 
        Covariance matrix w/ Tr norm = scale * dim
    '''
    
    lamda = np.zeros(dim)
    lamda[lowcut:highcut] = 1

    grid = np.arange(0, dim)
    yy, xx = np.meshgrid(grid, grid)
    diff = xx - yy

    # sinusoidal part
    C_cos = np.zeros((dim, dim))
    for j in range(lowcut, highcut):
        C_cos += lamda[j] * np.cos(2 * np.pi * j * diff / dim)

    # exponential part
    C_exp = np.exp(((xx - dim) + (yy - dim)) / decay_coef ** 2)

    # final covariance matrix
    C = C_cos * C_exp 
    C *= (scale * dim / np.trace(C))
    C += 1e-5 * np.eye(dim)
    return C

def sensilla_weights(num_weights, dim, lowcut, highcut, decay_coef=np.inf, scale=1, seed=None):
    """
    Generates random weights with tuning similar to mechanosensory 
    neurons found in insect halteres and wings.

    Parameters
    ----------

    num_weights: int
        Number of random weights

    dim : int
        dim of each random weight

    lowcut: int
        Low end of the frequency band. 

    highcut: int
        High end of the frequency band.
        
    decay_coef : float, default=np.inf
        controls the how fast the random features decay
        with default value, the weights do not decay
    
    seed : int, default=None
        Used to set the seed when generating random weights.
    
    Returns
    -------

    W : array-like of shape (num_weights, dim)
        Matrix of Random weights.
    """
    np.random.seed(seed)
    C = sensilla_covariance_matrix(dim, lowcut, highcut, decay_coef, scale)
    W = np.random.multivariate_normal(np.zeros(dim), cov=C, size=num_weights)
    return W


def V1_covariance_matrix(dim, size, spatial_freq, center, scale=1):
    """
    Generates the covariance matrix for Gaussian Process with non-stationary 
    covariance. This matrix will be used to generate random 
    features inspired from the receptive-fields of V1 neurons.

    C(x, y) = exp(|x - y|^2/2 * spatial_freq) * exp(|x - m|^2/ (2 * size)) * exp(|y - m|^2/ (2 * size))

    Parameters
    ----------

    dim : tuple of shape (2, 1)
        Dimension of random features.

    size : float
        Determines the size of the random weights 

    spatial_freq : float
        Determines the spatial frequency of the random weights  
    
    center : tuple of shape (2, 1)
        Location of the center of the random weights.

    scale: float, default=1
        Normalization factor for Tr norm of cov matrix

    Returns
    -------

    C : array-like of shape (dim[0] * dim[1], dim[0] * dim[1])
        covariance matrix w/ Tr norm = scale * dim[0] * dim[1]
    """

    x = np.arange(dim[0])
    y = np.arange(dim[1])
    yy, xx = np.meshgrid(y, x)
    grid = np.column_stack((xx.flatten(), yy.flatten()))

    a = squareform(pdist(grid, 'sqeuclidean'))
    b = la.norm(grid - center, axis=1) ** 2
    c = b.reshape(-1, 1)
    C = np.exp(-a / (2 * spatial_freq ** 2)) * np.exp(-b / (2 * size ** 2)) * np.exp(-c / (2 * size ** 2))
    C += 1e-5 * np.eye(dim[0] * dim[1])
    C *= (scale * dim[0] * dim[1] / np.trace(C))
    return C


def V1_weights(num_weights, dim, size, spatial_freq, center=None, scale=1, seed=None):
    """
    Generate random weights inspired by the tuning properties of the 
    neurons in Primary Visual Cortex (V1).

    If a value is given for the center, all generated weights have the same center
    If value is set to None, the centeres randomly cover the RF space

    Parameters
    ----------

    num_weights : int
        Number of random weights

    dim : tuple of shape (2,1)
        dim of each random weights
    
    size : float
        Determines the size of the random weights

    spatial_freq : float
        Determines the spatial frequency of the random weights 

    center: tuple of shape (2, 1), default = None
        Location of the center of the random weights
        With default value, the centers uniformly cover the RF space

    scale: float, default=1
        Normalization factor for Tr norm of cov matrix

    seed : int, default=None
        Used to set the seed when generating random weights.

    Returns
    -------

    W : array-like of shape (num_weights, dim[0] * dim[1])
        Matrix of random weights

    """
    np.random.seed(seed)
    if center == None: # centers uniformly cover the receptive field space
        W = np.zeros((num_weights, dim[0] * dim[1]))
        centers = np.random.randint((dim[0], dim[1]), size=(num_weights, 2))
        for i, c in enumerate(centers):
            C = V1_covariance_matrix(dim, size, spatial_freq, c, scale)
            W[i] = np.random.multivariate_normal(mean=np.zeros(dim[0] * dim[1]), cov=C, size=1)
 
    elif center != None:
        C = V1_covariance_matrix(dim, size, spatial_freq, center, scale)
        W = np.random.multivariate_normal(mean=np.zeros(dim[0] * dim[1]), cov=C, size=num_weights)
    return W


def classical_covariance_matrix(dim, scale=1):
    """
    Generates the covariance matrix for Gaussian Process with identity covariance. 
    This matrix will be used to generate random weights that are traditionally used 
    in kernel methods.

    C(x, y) = \delta_{xy}

    Parameters
    ----------

    dim: int or tuple (2, 1)
        dimension of each weight
        int for time-series, tuple for images 

    scale: float, default=1
        Normalization factor for Tr norm of cov matrix

    Returns
    -------

    C : array-like of shape (dim, dim) or (dim[0] * dim[1], dim[0] * dim[1])
        covariance matrix w/ Tr norm = scale * dim[0] * dim[1]
    """
    if type(dim) is tuple:
        C = np.eye(dim[0] * dim[1]) * scale

    elif type(dim) is int:
        C = np.eye(dim) * scale
    return C


def classical_weights(num_weights, dim, scale=1, seed=None):
    """"
    Generates classical random weights with identity covariance. W ~ N(0, 1).

    Parameters
    ----------

    num_weights : int
        Number of random weights

    dim : int
        dimension of each random weight

    scale : float, default=1
        Normalization factor for Tr norm of cov matrix
    
    seed : int, default=None
        Used to set the seed when generating random weights.

    Returns
    -------

    W : array-like of shape (num_weights, dim) or (num_weights, dim[0] * dim[1])
        Matrix of random weights.
    """
    C = classical_covariance_matrix(dim, scale)
    if type(dim) is tuple:
        W = np.random.multivariate_normal(mean=np.zeros(dim[0] * dim[1]), cov=C, size=num_weights)
    elif type(dim) is int:
        W = np.random.multivariate_normal(mean=np.zeros(dim), cov=C, size=num_weights)
    return W