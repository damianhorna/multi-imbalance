from imblearn.metrics import geometric_mean_score


def gmean_score(y_test, y_pred, correction: float = 0.001) -> float:  # pragma no cover
    """
    Calculate geometric mean score

    :param y_test:
        numpy array with labels
    :param y_pred:
        numpy array with predicted labels
    :param correction:
        value that replaces 0 during multiplication to avoid zeroing the result
    :return:
        geometric_mean_score: float
    """
    return geometric_mean_score(y_test, y_pred, correction=correction)
