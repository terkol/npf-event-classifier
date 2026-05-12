import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def histograms(X, y):
    fig, axes = plt.subplots(10, 10, figsize=(18,10))
    for i,ax in enumerate(axes.ravel()):
        a = 0.7
        col = X.iloc[:,i]
        col_ii = col[y==3]
        col_ia = col[y==2]
        col_ib = col[y==1]
        ax.set_title(X.columns[i])
        xmin = np.min(col.values)
        xmax = np.max(col.values)
        ax.set_yscale("log")
        bins = np.linspace(xmin, xmax, 21)
        ax.hist(col_ii, bins=bins,alpha=a, color='red')
        ax.hist(col_ib, bins=bins,alpha=a, color='lime')
        ax.hist(col_ia, bins=bins,alpha=a, color='blue')
    plt.tight_layout()
    plt.show()

def histograms_binary(X, y):
    fig, axes = plt.subplots(10, 10, figsize=(18,10))
    for i,ax in enumerate(axes.ravel()):
        a = 0.7
        col = X.iloc[:,i]
        col_ib = col[y==1]
        col_nonevent = col[y==0]
        ax.set_title(X.columns[i])
        xmin = np.min(col.values)
        xmax = np.max(col.values)
        ax.set_yscale("log")
        bins = np.linspace(xmin, xmax, 11)
        ax.hist(col_nonevent, bins=bins, label='nonevent',alpha=1)
        ax.hist(col_ib, bins=bins, label='event',alpha=a)
    plt.tight_layout()
    plt.show()

def covariances(X, y):
    std = StandardScaler().fit_transform(X)
    X_std = pd.DataFrame(columns=X.columns, data=std)
    X_std['class2'] = y
    cov = X_std.cov()
    cov_y = cov['class2']
    high_cov = cov_y[abs(cov_y)>0.2]
    print(high_cov.index)

    fig, ax = plt.subplots(1,2, figsize=(14,6))
    ax[0].set_title('Correlation histogram')
    ax[0].hist(cov_y, bins=20)
    ax[0].set_ylabel('n')
    ax[0].set_xlabel("Correlation with 'class2'")
    ax[1].set_title('Correlation histogram')
    ax[1].grid()
    ax[1].scatter(list(range(len(cov_y))),cov_y)
    ax[1].set_ylabel("Correlation with 'class2'")
    ax[1].set_xlabel('Spot in intrinsic order')
    plt.show()


if __name__ == "__main__":
    path = Path(__file__).parent.parent / "data"

    train = pd.read_csv(path / 'train.csv')
    train = train.drop(columns=['date', 'partlybad']).set_index('id')
    train['class4'] = train['class4'].map({'II': 3, 'Ia': 2, 'Ib': 1, 'nonevent': 0})

    X = train.drop(columns='class4')
    y = train['class4']
    histograms(X, y)
    histograms_binary(X, y)
    covariances(X, y)
