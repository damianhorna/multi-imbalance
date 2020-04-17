from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from multi_imbalance.utils.data import construct_flat_2pc_df

sns.set_style('darkgrid')


def plot_number_and_2d_data(X, y, dataset_name) -> None:  # pragma no cover
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

    axs[0].set_xlabel("class")
    fig.suptitle(dataset_name, fontsize=16)

    sns.countplot(y, ax=axs[0], palette=p)
    X = pca.transform(X)
    df = construct_flat_2pc_df(X, y)
    sns.scatterplot(x='x1', y='x2', hue='y', style='y', data=df, alpha=0.7, ax=axs[1], legend='full', palette=p)
