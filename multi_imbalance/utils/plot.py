from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from multi_imbalance.utils.data import construct_flat_2pc_df

sns.set_style('darkgrid')


def plot_cardinality_and_2d_data(X, y, dataset_name='') -> None:  # pragma no cover
    """

    Parameters
    ----------
    X two dimensional numpy array
    y two dimensional numpy array
    dataset_name : str

    Returns None
    -------

    """
    n = len(Counter(y).keys())
    p = sns.color_palette("husl", n)

    pca = PCA(n_components=2)
    pca.fit(X)

    fig, axs = plt.subplots(ncols=2)
    fig.set_size_inches(12, 6)
    axs = axs.flatten()

    fig.suptitle(dataset_name, fontsize=16)

    sns.countplot(y, ax=axs[0], palette=p)
    X = pca.transform(X)
    df = construct_flat_2pc_df(X, y)
    sns.scatterplot(x='x1', y='x2', hue='y', style='y', data=df, alpha=0.7, ax=axs[1], legend='full', palette=p)

    axs[0].set_xlabel("class")
    axs[0].set_ylabel("cardinality")


def plot_visual_comparision_datasets(X1, y1, X2, y2, dataset_name1='', dataset_name2='') -> None:  # pragma no cover
    """

    Parameters
    ----------
    X1 two dimensional numpy array
    X2 two dimensional numpy array
    y1 two dimensional numpy array
    y2 two dimensional numpy array
    dataset_name1 : str
    dataset_name2 : str

    Returns None
    -------

    """
    n = len(Counter(y1).keys())
    p = sns.color_palette("husl", n)

    pca = PCA(n_components=2)
    pca.fit(X1)

    fig, axs = plt.subplots(ncols=2, nrows=2)
    fig.set_size_inches(16, 10)
    axs = axs.flatten()

    sns.countplot(y1, ax=axs[0], palette=p)
    transformed_X = pca.transform(X1)
    df = construct_flat_2pc_df(transformed_X, y1)
    sns.scatterplot(x='x1', y='x2', hue='y', style='y', data=df, alpha=0.7, ax=axs[1], legend='full', palette=p)

    sns.countplot(y2, ax=axs[2], palette=p)
    transformed_X2 = pca.transform(X2)
    df = construct_flat_2pc_df(transformed_X2, y2)
    sns.scatterplot(x='x1', y='x2', hue='y', style='y', data=df, alpha=0.7, ax=axs[3], legend='full', palette=p)

    axs[0].set_xlabel("class")
    axs[2].set_xlabel("class")
    axs[0].set_ylabel("cardinality")
    axs[2].set_ylabel("cardinality")
    axs[1].set_title(dataset_name1)
    axs[3].set_title(dataset_name2)