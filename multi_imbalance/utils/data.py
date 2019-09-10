from nptyping import Array
from sklearn.decomposition import PCA
import pandas as pd


def construct_flat_2pc_df(X: Array[float], y: Array) -> pd.DataFrame:
    """
    This function uses PCA to reduce X dimensions and creates dataframe with 2 principal component and labels
    for this data. In case of X with 2 dimension it skips the PCA part

    Parameters
    ----------
    X multi dimensional numpy array (at least 2 dimensions)
    y one dimensional numpy array with labels

    Returns dataframe with 3 columns x1 x2 and y and with number of rows equal to number of rows in X
    -------

    """
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

    y = pd.DataFrame({'y': y})
    X_df = pd.DataFrame(data=X, columns=['x1', 'x2'])

    df = pd.concat([X_df, y], axis=1)

    return df
