from imblearn.metrics import geometric_mean_score


def gmean_score(y_test, y_pred, correction: float = 0.001) -> float:  # pragma no cover
    """
    Calculate geometric mean score

    Parameters
    ----------
    y_test numpy array with labels
    y_pred numpy array with predicted labels
    correction value that replaces 0 during multiplication to avoid zeroing the result

    Returns
    geometric_mean_score: float
    -------

    """
    return geometric_mean_score(y_test, y_pred, correction=correction)
