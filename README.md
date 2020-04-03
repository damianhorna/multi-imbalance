[![Build Status](https://travis-ci.org/damian-horna/multi-imbalance.svg?branch=master)](https://travis-ci.org/damian-horna/multi-imbalance)
[![Documentation Status](https://readthedocs.org/projects/multi-imbalance/badge/?version=latest)](https://multi-imbalance.readthedocs.io/en/latest/?badge=latest)
# multi-imbalance
Multi-class imbalance is a common problem occurring in real-world supervised classifications tasks. While there has already been some research on the specialized methods aiming to tackle that challenging problem, most of them still lack coherent Python implementation that is simple, intuitive and easy to use.
multi-imbalance is a python package tackling the problem of multi-class imbalanced datasets in machine learning.
## Requirements
Tha package has been tested under python 3.7. Relies heavily on scikit-learn and typical scientific stack (numpy, scipy, pandas etc.).

## Installation
Just type in
```bash
pip install multi-imbalance
```

## Implemented algorithms
Our package includes implementation of such algorithms, as: 
* ECOC, 
* OVO, 
* MDO, 
* GlobalCS, 
* SOUP, 
* SOUP Bagging, 
* Multi-class Roughly Balanced Bagging
* SPIDER3.

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

For more examples please refer to https://multi-imbalance.readthedocs.io/en/latest/
