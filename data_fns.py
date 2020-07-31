''' BP: Feb 12, 2020
Functions for generating data to test random features classifier.
'''

import numpy as np
import scipy
from scipy import signal

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return signal.lfilter(b, a, data)

def highpass_filter(data, highcut, fs, order=2):
    nyq = fs/2
    high = highcut / nyq
    b, a = signal.butter(order, high, btype='high')
    return signal.filtfilt(b, a, data)

def lowpass_filter(data, lowcut, fs, order=2):
    nyq = fs/2
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    return signal.filtfilt(b, a, data)


def noisy_sin_signal(f_s=2000, dur=10, f_signal=40, sig_dur=0.05, a=1, seed=None):
    '''
    Generate a time series of random noise interspersed with periods where a sinusoidal signal is on
    
    Parameters
    ----------
    
    f_s = sampling frequency (Hz)
    dur = duration of the time series (s)
    f_signal = frequency of the signal (Hz)
    sig_dur = duration of the signal (s)
    a: 2x amplitude of the signal used for snr (between 0 and 1)
    if n_amp = 2 x noise_amplitude: a ** 2 + n_amp ** 2 = 1; snr = a**2/n_amp**2
    
    Returns
    -------
    time_series: an array of shape (f_s * dur, 1)
    labels: array with binary values (0, 1) shape (f_s * dur, )
    '''
    if seed is not None:
        np.random.seed(seed)
    
    if a < 0 or a > 1:
        raise ValueError('a should be between 0 and 1')
    n_amp = np.sqrt(1 - a ** 2) 
    
    t_points = np.arange(0, dur, 1/f_s)
    t_series = np.random.normal(0, 1, dur * f_s)
    label = np.zeros(len(t_series))
    
    d = int(sig_dur * f_s) # array len of signal
    p = np.arange(0, len(t_points), d) # array with points d apart
    idx = np.random.choice(p, int(dur / (2 * sig_dur)), replace=False)
    for i in idx:
        t = t_points[i:i + d]
        t_series[i:i + d] = np.sqrt(2) * a * np.sin(2 * np.pi * f_signal * t) + n_amp * np.random.randn(len(t))
        label[i:i + d] = 1
    return t_series, label


def data_matrix(series, label, N=40):
    ''' 
    From a time series, generate a n_sample x n_feature matrix using overlapping windows.
    
    Parameters
    ----------
    series: 1-D numpy array shape (M, 1)
    label: 1-D numpy array shape (M, 1)
    N: n_features of the data matrix
    
    Returns
    -------
    X: array of shape (M - N, N)
    y: array of shape (M - N, )
    '''
    X = np.array([series[i:i + N] for i in range(len(series) - N)])
    y = label[N:]
    return X, y                       


def data_matrix_non_ov(series, label, N=40):
    ''' 
    From a time series, generate a n_sample x n_feature matrix using non-overlapping windows.
    
    Parameters
    ----------
    series: 1-D numpy array shape (M, 1)
    label: 1-D numpy array shape (M, 1)
    N: n_features of the data matrix
    
    Returns
    -------
    X: array of shape (M - N, N)
    y: array of shape (M - N, )
    '''
    X = np.array([series[N * i:N * (i + 1)] for i in range(int(len(series)/N))])
    y = label[N-1::N]
    return X, y 

def pure_sine_dft_v1(nPoints, fs, k, sig_dur, a, seed=None):
    ''' 
    Sample DFT matrix to generate a data matrix for classification. The positive examples are pure tone 
    sinusoids with k cycles in sig_dur while the negative examples are white noise N(0,1).
    
    Parameters
    ----------
    nPoints: total number of data points in the data matrix.
    fs: sampling frequency (Hz)
    sig_dur: duration of each data point (s)
    k: number of sinusoidal cycles in the duration of data point
    a: SNR of the signal
    seed: random state
    
    Returns
    -------
    X: array of shape (nPoints) x (fs * sig_dur)
    y: array with binary values (-1,1) of shape (nPoints,)
    '''
    
    if seed is not None:
        np.random.seed(seed)

    N = int(fs * sig_dur)
    noise_amp = np.sqrt(1 - a ** 2)
    
    # dft matrix
    A = scipy.linalg.dft(N, scale=None)
    
    # positive examples
    n_pos = int(nPoints/2)
    c = np.zeros((N, n_pos), dtype=complex)
    phi = np.random.uniform(-np.pi, np.pi, (n_pos,)) # randomly sample the phase
    c[k] = np.e ** (1j * phi) 
    X_pos = np.sqrt(2) * a * (A @ c).T.real + noise_amp * np.random.normal(size=(n_pos, N))
    
    # negative examples 
    X_neg = np.random.normal(size=(n_pos, N))
    
    # concatenate
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(n_pos) , np.ones(n_pos) * -1))
    
    return X, y


def pure_sine_dft_v2(nPoints, fs, k, sig_dur, a, seed=None):  
    ''' Same as above except the everything is generated from the dft matrix'''
    
    if seed is not None:
        np.random.seed(seed)

    N = int(fs * sig_dur)
    noise_amp = np.sqrt(1 - a ** 2)

    # dft matrix
    A = scipy.linalg.dft(N, scale=None)

    # positive examples
    n_pos = int(nPoints/2)
    c = np.zeros((N, n_pos), dtype=complex)
    phi = np.random.uniform(-np.pi, np.pi, (n_pos,)) # randomly sample the phase
    c[k] = np.e ** (1j * phi) 
    X_pos = np.sqrt(2) * a * (A @ c).T.real 

    # noise for positive egs
    phi_noise = np.random.uniform(-np.pi, np.pi, (n_pos, N))
    c_noise = np.random.normal(size=(n_pos, N)).astype(complex) * np.e ** (1j * phi_noise)
    noise = (c_noise @ A).real
    noise = noise_amp * noise / np.std(noise, axis=1).reshape(-1, 1)
    X_pos += noise

    # negative examples
    phi_neg = np.random.uniform(-np.pi, np.pi, (n_pos, N))
    c_neg = np.random.normal(size=(n_pos, N)).astype(complex) * np.e ** (1j * phi_neg)
    X_neg = (c_neg @ A).real
    X_neg = X_neg / np.std(X_neg, axis=1).reshape(-1, 1)

    # concatenate
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(n_pos) , np.ones(n_pos) * -1))
    
    return X, y

def pure_sine_dft(nPoints, fs, k, sig_dur, a, seed=None):  
    ''' Same as above except everything is generated from a gaussian process. Before, 
    there was a uniform distribution for the phase'''

    if seed is not None:
        np.random.seed(seed)

    N = int(fs * sig_dur)
    noise_amp = np.sqrt(1 - a ** 2)

    # dft matrix
    A = scipy.linalg.dft(N, scale=None)

    # positive examples
    n_pos = int(nPoints/2)
    c = np.zeros((N, n_pos), dtype=complex)
    rand = np.random.normal(loc=0, scale=1, size=(n_pos, 2)).view(np.complex).flatten()
    c[k] = rand / np.abs(rand)
    X_pos = np.sqrt(2) * a * (A @ c).T.real 

    # noise for positive egs
    rand = np.random.normal(loc=0, scale=1, size=(N, n_pos, 2)).view(np.complex).squeeze(axis=2)
    noise = (rand.T @ A).real
    noise = noise_amp * noise / np.std(noise, axis=1).reshape(-1, 1)
    X_pos += noise

    # negative examples
    rand = np.random.normal(loc=0, scale=1, size=(N, n_pos, 2)).view(np.complex).squeeze(axis=2)
    X_neg = (rand.T @ A).real
    X_neg = X_neg / np.std(X_neg, axis=1).reshape(-1, 1)

    # concatenate
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(n_pos) , np.ones(n_pos) * -1))
    
    return X, y


def XOR_data(nPoints, fs, k1, k2, sig_dur, a, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = int(fs * sig_dur)
    noise_amp = np.sqrt(1 - a ** 2)

    #dft matrix
    A = scipy.linalg.dft(N, scale=None)

    # positive examples
    n_pos = int(nPoints/2)
    c = np.zeros((N, n_pos), dtype='complex')
    rand = np.random.normal(loc=0, scale=1, size=(int(n_pos/ 2), 2)).view(np.complex).flatten()
    rand /= np.abs(rand)
    c[k1, :int(n_pos/2)] = rand

    rand = np.random.normal(loc=0, scale=1, size=(int(n_pos/ 2), 2)).view(np.complex).flatten()
    rand /= np.abs(rand)
    c[k2, int(n_pos/2):] = rand
    X_pos = np.sqrt(2) * a * (A @ c).T.real

    # noise for positive egs
    rand = np.random.normal(loc=0, scale=1, size=(N, n_pos, 2)).view(np.complex).squeeze(axis=2)
    rand /= np.abs(rand)
    noise = np.sqrt(2) * noise_amp / np.sqrt(N - 1) * (rand.T @ A).real
    X_pos += noise

    # negative egs
    n_neg = int(nPoints/2)

    # mixed egs
    c = np.zeros((N, int(n_neg/2)), dtype='complex')
    rand = np.random.normal(loc=0, scale=1, size=(1, int(n_neg/2), 2)).view(np.complex).squeeze(axis=2)
    rand /= np.abs(rand)
    c[[k1, k2]] = rand
    X_mixed = a * (A @ c).T.real

    # noise for mixed egs
    rand = np.random.normal(loc=0, scale=1, size=(N, int(n_neg/2), 2)).view(np.complex).squeeze(axis=2)
    rand /= np.abs(rand)
    noise = np.sqrt(2) * noise_amp / np.sqrt(N - 2) * (rand.T @ A).real
    X_mixed += noise

    # noise as negative egs
    c = np.random.normal(loc=0, scale=1, size=(N, int(n_neg/2), 2)).view(np.complex).squeeze(axis=2)
    c /= np.abs(c)
    X_noise = np.sqrt(2) * (A @ c).T.real / np.sqrt(N)
    
    X_neg = np.row_stack((X_mixed, X_noise))
        
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(n_pos), np.ones(n_neg) * -1))
    
    return X, y