from sklearn.decomposition import PCA
import pandas as pd


def construct_flat_2pc_df(X, y) -> pd.DataFrame:
    """
    This function takes two dimensional X and one dimensional y arrays, concatenates and returns them as data frame

    Parameters
    ----------
    X two dimensional numpy array
    y one dimensional numpy array with labels

    Returns data frame with 3 columns x1 x2 and y and with number of rows equal to number of rows in X
    -------

    """

    y = pd.DataFrame({'y': y})
    X_df = pd.DataFrame(data=X, columns=['x1', 'x2'])

    df = pd.concat([X_df, y], axis=1)

    return df
