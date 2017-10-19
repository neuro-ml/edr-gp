import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from edrgp.regression import GaussianProcessRegressor
from edrgp.edr import EffectiveDimensionalityReduction
from edrgp.datasets import get_gaussian_inputs, get_tanh_targets
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
sns.set()

PALETTE = sns.color_palette()
MARKERS = ['o', 's', 'v']
MARKER_SIZE = 12
CMAP = sns.diverging_palette(220, 20, s=99, as_cmap=True)


def get_2d_data():
    '''Returns a simple 2D dataset'''
    X = get_gaussian_inputs(
        eig_values=[1, 0.3], sample_size=500,
        eig_vectors=np.array([[1, 1], [-1, 1]]))
    X -= X.mean(0)
    y = get_targets(X)
    return X, y


def get_targets(X, **kwargs):
    return get_tanh_targets(X, [0.5, 0.5], **kwargs)


def plot_data(X, y):
    plt.figure(figsize=[8, 5])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP)
    plt.colorbar(label='Target variable')
    plt.title('The dataset')
    plt.xlabel('Feature 1', fontsize=16)
    plt.ylabel('Feature 2', fontsize=16)


def plot_dr_component(X, y, dr, title):
    plt.figure(figsize=[12, 5])
    plt.subplot(1, 2, 1)
    # plot dataset
    plt.plot(*X.T, '.', label='Training sample')
    plt.xlabel('Feature 1', fontsize=16)
    plt.ylabel('Feature 2', fontsize=16)
    # plot direction (preserving figure limits)
    xlim = plt.xlim()
    ylim = plt.ylim()
    x1, x2 = dr.components_[0] * max(*xlim, *ylim)
    plt.plot([-x1, x1], [-x2, x2], lw=4, label='The selected combination')
    plt.xlim(xlim)
    plt.ylim(ylim)

    # add 3 artificial points
    X_sample = np.array([[-2, 1], [-1.3, 2.3], [2.3, -0.7]])
    y_sample = get_targets(X_sample, noise_std=0)
    # plot these points and their projections
    labels = ['Original features', 'Features after projection', 'Projection']
    for i, (x, marker) in enumerate(zip(X_sample, MARKERS)):
        x_proj = dr.inverse_transform(dr.transform(x[np.newaxis, :]))[0]
        plt.plot(*x, marker, c=PALETTE[3], label=labels[0], ms=MARKER_SIZE)
        plt.plot(*x_proj, marker, c=PALETTE[2], label=labels[1],
                 ms=MARKER_SIZE)
        plt.plot([x[0], x_proj[0]], [x[1], x_proj[1]], '--', c=PALETTE[2],
                 label=labels[2])
        labels = ['_', '_', '_']
    plt.legend(loc='best')

    # add a plot with projected data
    plt.subplot(1, 2, 2)
    plt.suptitle(title, fontsize=18)
    plt.scatter(dr.transform(X), y)
    for i in range(len(X_sample)):
        plt.plot(dr.transform(X_sample[i:i + 1]), y_sample[i:i + 1],
                 marker=MARKERS[i], ms=MARKER_SIZE, c=PALETTE[2])
    plt.xlabel('The found linear combination ', fontsize=16)
    plt.ylabel('The target variable', fontsize=16)
    mi = mutual_info_regression(dr.transform(X), y)[0]
    txt = ('Mutual information beween the target and '
           'the found linear combination is {:.3f}'.format(mi))
    print(txt)


def plot_explained_variance(X, y):
    # build full PCA to estimate all variance ratios
    pca = PCA(n_components=2)
    pca.fit(X)
    pca_variance = pca.explained_variance_ratio_
    # the same operation for EDR
    edr = EffectiveDimensionalityReduction(
        GaussianProcessRegressor(), PCA(n_components=2), True)
    edr.fit(X, y)
    edr_variance = edr.dr_transformer_.explained_variance_ratio_
    # bars for PCA
    plt.figure(figsize=[12, 5])
    plt.subplot(1, 2, 1)
    plt.title('PCA - explained features variance')
    sns.barplot(x=[1, 2], y=pca_variance)
    plt.ylim([0, 1])
    plt.xlabel('Component number')
    # bars for EDR-GP
    plt.subplot(1, 2, 2)
    plt.title('EDR-GP - explained functional variance')
    sns.barplot(x=[1, 2], y=edr_variance)
    plt.xlabel('Component number')


if __name__ == "__main__":
    X, y = get_2d_data()
    plot_data(X, y)

    # plot DR
    pca = PCA(n_components=1)
    pca.fit(X)
    plot_dr_component(X, y, pca, 'Principal Component Analysis')

    # plot EDR-GP
    edr = EffectiveDimensionalityReduction(
        GaussianProcessRegressor(), PCA(n_components=1), True)
    edr.fit(X, y)
    plot_dr_component(X, y, edr, 'Effective dimensionality reduction')

    # compare explained variance ratio
    plot_explained_variance(X, y)
