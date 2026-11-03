# Exploratory notebooks

Scratch notebooks, not needed to reproduce the paper (see `notebooks/reports/`).

Some of them use `dask` for parallel hyperparameter search, which is not in the top-level `requirements.txt`. To run them, install it in the project environment:

```
pip install "dask[distributed]"
```

Notebooks that import it: `MNIST_classification`, `estimator_test`, `fashion-MNIST_clf`, `frequency_XOR`, `frequency_detection`, `haltere_rand_features_test`, `kmnist_clf`.

Running them creates a `dask-worker-space/` directory here. It is a runtime artifact and can be deleted.
