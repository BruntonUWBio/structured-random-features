Structured random receptive fields
==============================

This code generates the analyses in the paper "Structured random receptive fields enable informative sensory encodings" 
([biorxiv preprint](https://www.biorxiv.org/content/10.1101/2021.09.09.459651)), published in PLoS Comp Bio.

__Summary__: In this work, we describe an approach to model receptive fields
of biological neurons as samples from Gaussian Processes (GP).
We successfully model receptive fields datasets in two disparate 
sensory regions, insect mechanosensors and mammalian primary visual cortex (V1).
Using our models, we also show a novel initialization scheme that improves
the learning efficiency of artificial neural networks (ANNs).

![framework](network.png)

Steps to set up the directory:
------------

The steps below use either [uv](https://docs.astral.sh/uv/) or plain pip. We have tested Python 3.10.

1. Do a clean download of the repository. Go inside the repo.
2. Make a virtual environment, unless you want to install into the base environment.

	| uv | pip |
	|---|---|
	| `uv venv --python 3.10` | `python3 -m venv .venv` |

3. Activate the new environment
	```
	source .venv/bin/activate
	```
4.  Install the package and the notebook tools.

	| uv | pip |
	|---|---|
	| `uv pip install -e .` | `pip install -e .` |
	| `uv pip install -r requirements.txt` | `pip install -r requirements.txt` |

5.  Download all the datasets and preprocess.

	| uv | pip |
	|---|---|
	| `make data PIP="uv pip"` | `make data` |

6.  Run the notebooks in *notebooks/reports* folder.

__Choosing a torch build__: install the desired torch and torchvision *before* step 4, since
`requirements.txt` otherwise installs the default build. The later `-e .` install leaves them untouched.
```
# pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or .../cpu
# uv
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
With uv you can instead pass its backend flag at install time: `uv pip install -e . --torch-backend cpu`
(other values: `cu126`, `auto`, ...).

Data sources
------------
The data for our analyses can be found in this [Google drive folder](https://drive.google.com/drive/folders/1XLfnikdNjnUTT87RBCMGJFa42N-ZfHKO?usp=sharing). 

Replicating our published findings
------------
- The source code is in the */structured_random_features* folder. The jupyter notebooks in */notebooks/reports* 
folder show the analyses used in the paper. 

- Notebooks 1--5 show how we fit our covariance models to biological data. We 
plot these results in Notebooks 16.0--16.4.

- Notebooks 6 and 7 show the results on the time-series classification tasks. We
plot these results in Notebook 16.5.

- Notebooks 8--11 show the results on the digit classification tasks. We plot these
results in Notebook 16.6.

- Notebooks 12--15 show our classification performance when initializing networks 
with structured random features. We plot these results in Notebook 16.7 and 16.8

- Notebooks 16.xx are used to plot the figures shown in the paper. They
assume that you have run the relevant notebooks and saved the results. 
