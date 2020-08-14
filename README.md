[![Build Status](https://travis-ci.org/damian-horna/multi-imbalance.svg?branch=master)](https://travis-ci.org/damian-horna/multi-imbalance)
[![codecov](https://codecov.io/gh/damian-horna/multi-imbalance/branch/master/graph/badge.svg)](https://codecov.io/gh/damian-horna/multi-imbalance)
[![Documentation Status](https://readthedocs.org/projects/multi-imbalance/badge/?version=latest)](https://multi-imbalance.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/multi-imbalance.svg)](https://badge.fury.io/py/multi-imbalance)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/multi-imbalance)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)

# multi-imbalance
Multi-class imbalance is a common problem occurring in real-world supervised classifications tasks. While there has already been some research on the specialized methods aiming to tackle that challenging problem, most of them still lack coherent Python implementation that is simple, intuitive and easy to use.
multi-imbalance is a python package tackling the problem of multi-class imbalanced datasets in machine learning.
## Requirements
Tha package has been tested under python 3.6, 3.7 and 3.8. It relies heavily on scikit-learn and typical scientific stack (numpy, scipy, pandas etc.).
Requirements include:
* numpy>=1.17.0,
* scikit-learn>=0.22.0,
* pandas>=0.25.1,
* pytest>=5.1.2,
* imbalanced-learn>=0.6.1
* IPython>=7.13.0,
* seaborn>=0.10.1,
* matplotlib>=3.2.1


## Installation
Just type in
```bash
pip install multi-imbalance
```

## Implemented algorithms
Our package includes implementation of such algorithms, as: 
* One-vs-One (OVO) and One-vs-all (OVA) ensembles [2],
* Error-Correcting Output Codes (ECOC) [1] with dense, sparse and complete encoding [9] ,
* Global-CS [4],
* Static-SMOTE [10],
* Mahalanobis Distance Oversampling [3],
* Similarity-based Oversampling and Undersampling Preprocessing (SOUP) [5],
* SPIDER3 cost-sensitive pre-processing [8].
* Multi-class Roughly Balanced Bagging (MRBB) [7],
* SOUP Bagging [6],

## Example usage
```python
from multi_imbalance.resampling.mdo import MDO

# Mahalanbois Distance Oversampling
mdo = MDO(k=9, k1_frac=0, seed=0)

# read the data
X_train, y_train, X_test, y_test = ...

# preprocess
X_train_resampled, y_train_resampled = mdo.fit_transform(np.copy(X_train), np.copy(y_train))

# train the classifier on preprocessed data
clf_tree = DecisionTreeClassifier(random_state=0)
clf_tree.fit(X_train_resampled, y_train_resampled)

# make predictions
y_pred = clf_tree.predict(X_test)
```

## Example usage with pipeline
At the moment, due to some sklearn's limitations the only way to use our **resampling** methods is to use the pipelines 
implemented in **imbalanced-learn**. It doesn't apply to **ensemble** methods.
```python
from imblearn.pipeline import Pipeline

X, y = load_arff_dataset('data/arff/new_ecoli.arff')
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mdo', MDO()),
    ('knn', KNN())
])

pipeline.fit(X_train, y_train)
y_hat = pipeline.predict(X_test)

print(classification_report(y_test, y_hat))
```

For more examples please refer to https://multi-imbalance.readthedocs.io/en/latest/ or check `examples` directory.

## For developers:
multi-imbalance follows sklearn's coding guideline: https://scikit-learn.org/stable/developers/contributing.html

We use pytest as our unit tests framework. To use it, simply run:
```bash
pytest
```

If you would like to check the code coverage:
```bash
coverage run -m pytest
coverage report -m # or coverage html
```

multi-imbalance uses reStructuredText markdown for docstrings. To build the documentation locally run:
```bash
cd docs
make html -B
```
and open `docs/_build/html/index.html`

if you add a new algorithm, we would appreciate if you include references and an example of use in `./examples` or docstrings.

## About
If you use multi-imbalance in a scientific publication, please consider including
citation to the following thesis:

```
@bachelorthesis{ MultiImbalance2020,
author = "Jacek Grycza, Damian Horna, Hanna Klimczak, Kamil Plucínski",
title = "Multi-imbalance:  Python package for multi-class imbalance learning",
school = "Poznan University of Technology",
address = "Poznan, Poland",
year = "2020",}
```

## References:

[1] Dietterich, T., and Bakiri, G. Solving multi-class learning problems via error-correcting output codes. Journal of Artificial Intelligence Research 2 (02 1995), 263–286.

[2] Fernández, A., López, V., Galar, M., del Jesus, M., and Herrera, F. Analysing the classification of imbalanced data-sets with multiple classes: Binarization techniques and ad-hoc approaches. Knowledge-Based Systems 42 (2013), 97 – 110.

[3] Abdi, L., and Hashemi, S. To combat multi-class imbalanced problems by means of over-sampling techniques. IEEE Transactions on Knowledge and Data Engineering 28 (January 2016), 238–251.

[4] Zhou, Z., and Liu, X. On multi-class cost-sensitive learning. In Proceedings of the 21st National Conference on Artificial Intelligence - Volume 1 (2006), AAAI’06, AAAI Press, pp. 567–572.

[5] Janicka, M., Lango, M., and Stefanowski, J. Using information on class interrelations to improve classification of multi-class imbalanced data: A new resampling algorithm. International Journal of Applied Mathematics and Computer Science 29 (December 2019).

[6] Lango, M., and Stefanowski, J. SOUP-Bagging: a new approach for multi-class imbalanced data classification. PP-RAI ’19: Polskie Porozumienie na Rzecz Sztucznej Inteligencji (2019).

[7] Lango, M., and Stefanowski, J. Multi-class and feature selection extensions of roughly balanced bagging for imbalanced data. J Intell Inf Syst 50 (2017), 97–127

[8] Wojciechowski, S., Wilk, S., and Stefanowski, J. An algorithm for selective preprocessing of multi-class imbalanced data. In Proceedings of the 10th International Conference on Computer Recognition Systems (05 2017), pp. 238–247.

[9] Kuncheva, L. Combining Pattern Classifiers: Methods and Algorithms. Wiley (2004).

[10] Fernández-Navarro, F., Hervás-Martínez, C., and Antonio Gutiérrez, P. A dynamic over-sampling procedure based on sensitivity for multi-class problems. Pattern Recognition, 44(8), 1821–1833 (2011).
