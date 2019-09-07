from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns


def plot_multi_dimensional_data(X, y, ax=None):
    """
    This function reduce quantity of dimensions to 2 principal components and prepare pretty scatter plot for your data
    :param X: multi dimensional numpy array (at least 2 dimensions)
    :param y: one dimensional numpy array with labels
    :param ax: optional parameter for subplots
    :return: None
    """

    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

    y = pd.DataFrame({'y': y})

    X_df = pd.DataFrame(data=X, columns=['x1', 'x2'])
    df = pd.concat([X_df, y], axis=1)

    sns.scatterplot(x="x1", y="x2", hue="y", data=df, alpha=1, ax=ax, legend=False)
