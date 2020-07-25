from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from multi_imbalance.ensemble.mrbbagging import MRBBagging

X, y = datasets.load_iris(return_X_y=True)
clf = make_pipeline(StandardScaler(), MRBBagging(30, DecisionTreeClassifier()))
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
print(cross_val_score(clf, X, y, cv=cv))
