from setuptools import find_packages, setup

setup(
    name='structured_random_features',
    packages=find_packages(),
    version='0.2.0',
    install_requires=['numpy', 'scipy', 'scikit-learn', 'scikit-image', 'torch',
                      'torchvision', 'pandas', 'matplotlib', 'seaborn', 'click',
                      'python-dotenv', 'tqdm', 'joblib'],
    description='We want to use ideas from random features literature to model neuronal tuning properties.',
    author='Biraj Pandey',
    license='MIT',
)
