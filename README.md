[![Build Status](https://travis-ci.org/damian-horna/multi-imbalance.svg?branch=master)](https://travis-ci.org/damian-horna/multi-imbalance)

# multi-imbalance

multi-imbalance is a python package tackling the problem of multi-class imbalanced datasets in machine learning.

## Requirements
Tha package has been tested under python 3.7. Relies heavily on scikit-learn and typical scientific stack (numpy, scipy, pandas etc.).

## Installation
Just type in
```bash
pip install multi-imbalance
```

## Implemented algorithms
    
1. SOUP, MDO
2. ECOC
3. Roughly Balanced Bagging
4. SPIDER3 algorithm implementation for selective preprocessing of multi-class imbalanced data sets, according to article:

    Wojciechowski, S., Wilk, S., Stefanowski, J.: An Algorithm for Selective Preprocessing
    of Multi-class Imbalanced Data. Proceedings of the 10th International Conference
    on Computer Recognition Systems CORES 2017

## Example usage
```python
from multi_imbalance.resampling.MDO import MDO

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
