import numpy as np
import scipy.linalg
import sklearn.utils as sk_utils

def frequency_detection(n_samples, fs, k, signal_duration, a, 
    random_state=None): 
    """
    Generate frequency detection task. The positive examples are pure
    sinusoids with additive gaussian noise and the negative examples are 
    white noise. The examples are generated using the DFT matrix.

    Parameters
    ----------

    n_samples : int
        Number of total examples

    fs : int
        Sampling rate of the signal
    
    k : int
        Column of the DFT matrix. The signal of the frequency is 
        int(k /  signal_duration)).

    signal_duration: float
        Length of the signal in seconds.

    a : float, 0 <= a <= 1
        Determines the SNR of the signal.
        SNR = a ** 2 / (1 - a ** 2)

    random_state : int
        Random state of the generated examples

    Returns
    -------
    X : (array-like) of shape (n_samples, n_features)
        Every row corresponds to an example with n_features components.
        n_features = fs * signal_duration

    y : (array-like) of shape (n_samples,)
        Target label (-1/1) for every example. 
    """

    if random_state is not None:
        np.random.seed(random_state)
    
    N = int(fs * signal_duration)
    noise_amplitude = np.sqrt(1 - a ** 2)

    # dft matrix
    A = scipy.linalg.dft(N, scale=None)

    # positive examples
    n_pos = int(n_samples / 2)
    c = np.zeros((N, n_pos), dtype='complex')
    rand = np.random.normal(loc=0, scale=1, size=(n_pos, 2)).view(np.complex).flatten()
    rand /= np.abs(rand)
    c[k] = rand
    X_pos = np.sqrt(2) * a * (A @ c).T.real

    # noise for positive egs
    rand = np.random.normal(loc=0, scale=1, size=(N, n_pos, 2)).view(np.complex).squeeze(axis=2)
    rand /= np.abs(rand)
    noise = np.sqrt(2) * noise_amplitude / np.sqrt(N - 1) * (A @ rand).T.real
    X_pos += noise

    # negative egs
    n_neg = int(n_samples / 2)
    c = np.random.normal(loc=0, scale=1, size=(N, n_neg, 2)).view(np.complex).squeeze(axis=2)
    c /= np.abs(c)
    X_neg = np.sqrt(2) * (A @ c).T.real / np.sqrt(N)

    # concatenate and shuffle
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(n_pos), np.ones(n_neg) * -1))
    X, y = sk_utils.shuffle(X, y)

    return X, y


def frequency_XOR(n_samples, fs, k1, k2, signal_duration, a, 
    random_state=None, shuffle=True):
    """
    Generates a frequency XOR task. The positive eg are single
    frequency sinusoids (2 frequencies) with additive gaussian noise. 
    The negative eg are mixed sinusoids or white noise.

    Parameters
    ----------

    n_samples : int
        Number of total examples

    fs : int
        Sampling rate of the signal
    
    k1 : int
        Column of the DFT matrix. The frequency of the signal is 
        int(k1 /  signal_duration)).

    k2 : int
        Column of the DFT matrix. The frequency of the signal is
        int(k2 /  signal_duration)). 

    signal_duration: float
        Length of the signal in seconds.

    a : float, 0 <= a <= 1
        Determines the SNR of the signal.
        SNR = a ** 2 / (1 - a ** 2)

    random_state : int
        Random state of the generated examples
    
    shuffle : Bool, default=True
        Shuffle data if True.
    
    Returns
    -------

    X : (array-like) of shape (n_samples, n_features)
        Every row corresponds to an example with n_features components.
        n_features = fs * signal_duration

    y : (array-like) of shape (n_samples,)
        Target label (-1/1) for every example. 
    """

    if random_state is not None:
        np.random.seed(random_state)

    N = int(fs * signal_duration)
    noise_amplitude = np.sqrt(1 - a ** 2)

    #dft matrix
    A = scipy.linalg.dft(N, scale=None)

    # positive examples
    n_pos = int(n_samples/2)
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
    noise = np.sqrt(2) * noise_amplitude / np.sqrt(N - 1) * (A @ rand).T.real
    X_pos += noise

    # negative egs
    n_neg = int(n_samples/2)

    # mixed egs
    c = np.zeros((N, int(n_neg/2)), dtype='complex')
    rand = np.random.normal(loc=0, scale=1, size=(1, int(n_neg/2), 2)).view(np.complex).squeeze(axis=2)
    rand /= np.abs(rand)
    c[[k1, k2]] = rand
    X_mixed = a * (A @ c).T.real

    # noise for mixed egs
    rand = np.random.normal(loc=0, scale=1, size=(N, int(n_neg/2), 2)).view(np.complex).squeeze(axis=2)
    rand /= np.abs(rand)
    noise = np.sqrt(2) * noise_amplitude / np.sqrt(N - 2) * (A @ rand).T.real
    X_mixed += noise

    # noise as negative egs
    c = np.random.normal(loc=0, scale=1, size=(N, int(n_neg/2), 2)).view(np.complex).squeeze(axis=2)
    c /= np.abs(c)
    X_noise = np.sqrt(2) * (A @ c).T.real / np.sqrt(N)
    
    X_neg = np.row_stack((X_mixed, X_noise))
    
    # concatenate and shuffle
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(n_pos), np.ones(n_neg) * -1))
    if shuffle is True:
        X, y = sk_utils.shuffle(X, y)
    return X, y


def load_mnist(data_path='./data/mnist'):
    """
    Loads the MNIST handwritten digits data. There are 60k training
    images and 10k testing images. Each image is (28, 28). The digits 
    are from 0-10.

    Parameters
    ----------
    data_path : string, default='./data/MNIST/'
        Path to the MNIST folder.
    
    Returns
    -------
    train : (array-like) of shape (60000, 784)
        Training data

    train_labels : (array-like) of shape (60000,)
        Training labels

    test : (array-like) of shape (10000, 784)
        Testing data

    test_labels : (array-like) of shape (10000,)
        Test data labels
    """
    from mnist import MNIST
    mndata = MNIST(data_path)
    train, train_labels = map(np.array, mndata.load_training())
    test, test_labels = map(np.array, mndata.load_testing())
    train = train/255
    test = test/255
    return train, train_labels, test, test_labels

def unpickle(file):
    """
    Open a pickle file to a dict. This is specifically for cifar_10 data. 

    Parameters
    ----------

    file : string
        Path of the pickle file.
    
    Returns
    -------

    dict : dict
        Pickle file as a dictionary
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar_10(data_path='./data/cifar_10/', grayscale=True):
    """
    Loads the CIFAR-10 dataset. There are 50k training images and 10k
    test images. There are 10 categories. Each image is (32, 32, 3). 

    Parameters
    ----------

    file : string
        Path to the cifar10 folder

    grayscale : bool, default=True
        Converts the color image into grayscale using Luma coding

    Returns
    -------
    train : (array-like) of shape (50000, n_features)
        Training data

    train_labels : (array-like) of shape (50000, n_features)
        Training labels

    test : (array-like) of shape (10000, n_features)
        Testing data

    test_labels : (array-like) of shape (10000,)
        Test data labels
    """
    train = np.empty((0, 3072))
    train_labels = []
    for i in range(5):
        batch = unpickle(data_path + 'data_batch_%s' % str(i + 1))
        train = np.row_stack((train, batch[b'data']))
        train_labels = np.concatenate((train_labels, batch[b'labels']))

    test_batch = unpickle(data_path + '/test_batch')
    test = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])

    if grayscale is True:
        train = 0.2989 * train[:, :1024] + 0.5870 * train[:, 1024:2048] + 0.1140 * train[:, -1024:]
        test = 0.2989 * test[:, :1024] + 0.5870 * test[:, 1024:2048] + 0.1140 * test[:, -1024:]
    return train, train_labels, test, test_labels

def load_kmnist(data_path='./data/kmnist/'):
    """ 
    Loads the Kuzushiji-MNIST dataset. The dataset is downloaded in the
    numpy version.

    Parameters
    ----------
    
    data_path : string
        Path to the kmnist data

    Returns
    -------
    train : (array-like) of shape (50000, n_features)
        Training data

    train_labels : (array-like) of shape (50000, n_features)
        Training labels

    test : (array-like) of shape (10000, n_features)
        Testing data

    test_labels : (array-like) of shape (10000,)
        Test data labels
    """
    train = np.load(data_path + 'kmnist-train-imgs.npz')['arr_0']
    train_labels = np.load(data_path + 'kmnist-train-labels.npz')['arr_0']
    test = np.load(data_path + 'kmnist-test-imgs.npz')['arr_0']
    test_labels = np.load(data_path + 'kmnist-test-labels.npz')['arr_0']

    train = train.reshape(-1, 784)
    test = test.reshape(-1, 784)
    train = train /255
    test = test /255

    return train, train_labels, test, test_labels

def download_omniglot(data_path='./data'):

    """
    Downloads the omniglot dataset. The dataset contains images of 1623 
    different characters with 20 examples of each character. Each image
    is of size (105, 105). We will load only the training set with 964 characters. It's the "background_images" in the original dataset.
    The images are saved in the datapath in a dict with labels 'data'
    and 'labels.'

    Parameters
    ----------

    data_path = string  
        Path to download data into

    Returns
    -------
    None

    """
    import torchvision as tv
    from torch.utils.data import DataLoader
    import pickle

    dataset = tv.datasets.Omniglot(root=data_path, download=True, 
    transform=tv.transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(dataloader))[0].numpy().squeeze()
    labels = next(iter(dataloader))[1].numpy()
    dict = {'data': data, 'labels': labels}

    with open(data_path + '/omniglot-py/omniglot.pickle', 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_omniglot(data_path='./data/omniglot-py/'):
    """
    Loads the omniglot dataset. The dataset contains images of 1623 
    different characters with 20 examples of each character. Each image
    is of size (105, 105). We will load only the training set with 964 characters. Download the data first with `download_omniglot` before
    using this function. 

    Parameters
    ----------

    data_path = string
        Path to the `omniglot-py` data folder

    Returns
    -------
    data : (array-like) of shape (964 * 20, 11025)
        Training data

    label : (array-like) of shape (964 * 20,)
        Target labels
    """
    import pickle

    with open(data_path + 'omniglot.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    return data_dict['data'], data_dict['labels']




