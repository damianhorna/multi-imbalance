.. multi-imbalance documentation master file, created by
   sphinx-quickstart on Fri Apr  3 12:53:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to multi-imbalance's documentation!
===========================================

Multi-class imbalance is a common problem occurring in real-world supervised classifications tasks. While there has already been some research on the specialized methods aiming to tackle that challenging problem, most of them still lack coherent Python implementation that is simple, intuitive and easy to use. multi-imbalance is a python package tackling the problem of multi-class imbalanced datasets in machine learning.

Installation:
^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install multi-imbalance

Example of code:
^^^^^^^^^^^^^^^^

.. code-block:: python

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

.. toctree::
   :maxdepth: 4
   :caption: Docstring

   source/docstring/modules

.. toctree::
       :maxdepth: 4
       :caption: License

       license
