[![Build Status](https://travis-ci.org/damian-horna/multi-imbalance.svg?branch=master)](https://travis-ci.org/damian-horna/multi-imbalance)
[![Documentation Status](https://readthedocs.org/projects/multi-imbalance/badge/?version=latest)](https://multi-imbalance.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/multi-imbalance.svg)](https://badge.fury.io/py/multi-imbalance)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)

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
* ECOC [1],
* OVO [2],
* MDO [3],
* GlobalCS [4], 
* SOUP [5],
* SOUP Bagging [6],
* Multi-class Roughly Balanced Bagging [7],
* SPIDER3 [8].

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

[1] Dietterich, T., and Bakiri, G. Solving multi-class learning problems via error-correcting
output codes. Journal of Artificial Intelligence Research 2 (02 1995), 263–286.

[2] Fernández, A., López, V., Galar, M., del Jesus, M., and Herrera, F. Analysing
the classification of imbalanced data-sets with multiple classes: Binarization techniques and
ad-hoc approaches. Knowledge-Based Systems 42 (2013), 97 – 110.

[3] Abdi, L., and Hashemi, S. To combat multi-class imbalanced problems by means of
over-sampling techniques. IEEE Transactions on Knowledge and Data Engineering 28
(January 2016), 238–251.

[4] Zhou, Z., and Liu, X. On multi-class cost-sensitive learning. In Proceedings of the 21st
National Conference on Artificial Intelligence - Volume 1 (2006), AAAI’06, AAAI Press,
pp. 567–572.

[5] Janicka, M., Lango, M., and Stefanowski, J. Using information on class interrelations
to improve classification of multi-class imbalanced data: A new resampling algorithm.
International Journal of Applied Mathematics and Computer Science 29 (December 2019).

[6] Lango, M., and Stefanowski, J. SOUP-Bagging: a new approach for multi-class
imbalanced data classification. PP-RAI ’19: Polskie Porozumienie na Rzecz Sztucznej
Inteligencji (2019).

[7] Lango, M., and Stefanowski, J. Multi-class and feature selection extensions of roughly
balanced bagging for imbalanced data. J Intell Inf Syst 50 (2017), 97–127

[8] Wojciechowski, S., Wilk, S., and Stefanowski, J. An algorithm for selective
preprocessing of multi-class imbalanced data. In Proceedings of the 10th International
Conference on Computer Recognition Systems (05 2017), pp. 238–247.