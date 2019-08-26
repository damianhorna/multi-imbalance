import pandas as pd

from multi_imbalance.resampling.SOUP import SOUP

# TODO move it to correct directory

# TODO replace it by correct file in repository
ecoli_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'
df = pd.read_csv(ecoli_url, delim_whitespace=True, header=None,
                 names=['name', '1', '2', '3', '4', '5', '6', '7', 'class'])

X, y = df.iloc[:, 1:8].to_numpy(), df['class'].to_numpy()

clf = SOUP()
clf.fit(X, y)
