from imblearn.metrics import geometric_mean_score


def gmean_score(y_test, y_pred, correction=0.001):  # pragma no cover
    return geometric_mean_score(y_test, y_pred, correction=correction)
