"""Plotting support for the UC Irvine course on 'ML & Statistics for Physicists'

Matplotlib must be installed to import this module.
"""

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import colorConverter, ListedColormap
from matplotlib.collections import EllipseCollection

import scipy.stats


def draw_ellipses(w, mu, C, nsigmas=2, color='red', outline=None, filled=True, axis=None):
    """Draw a collection of ellipses.

    Uses the low-level EllipseCollection to efficiently draw a large number
    of ellipses. Useful to visualize the results of a GMM fit via
    GMM_pairplot() defined below.

    Parameters
    ----------
    w : array
        1D array of K relative weights for each ellipse. Must sum to one.
        Ellipses with smaller weights are rendered with greater transparency
        when filled is True.
    mu : array
        Array of shape (K, 2) giving the 2-dimensional centroids of
        each ellipse.
    C : array
        Array of shape (K, 2, 2) giving the 2 x 2 covariance matrix for
        each ellipse.
    nsigmas : float
        Number of sigmas to use for scaling ellipse area to a confidence level.
    color : matplotlib color spec
        Color to use for the ellipse edge (and fill when filled is True).
    outline : None or matplotlib color spec
        Color to use to outline the ellipse edge, or no outline when None.
    filled : bool
        Fill ellipses with color when True, adjusting transparency to
        indicate relative weights.
    axis : matplotlib axis or None
        Plot axis where the ellipse collection should be drawn. Uses the
        current default axis when None.
    """
    # Calculate the ellipse angles and bounding boxes using SVD.
    U, s, _ = np.linalg.svd(C)
    angles = np.degrees(np.arctan2(U[:, 1, 0], U[:, 0, 0]))
    widths, heights = 2 * nsigmas * np.sqrt(s.T)
    # Initialize colors.
    color = colorConverter.to_rgba(color)
    if filled:
        # Use transparency to indicate relative weights.
        ec = np.tile([color], (len(w), 1))
        ec[:, -1] *= w
        fc = np.tile([color], (len(w), 1))
        fc[:, -1] *= w ** 2
    # Data limits must already be defined for axis.transData to be valid.
    axis = axis or plt.gca()
    if outline is not None:
        axis.add_collection(EllipseCollection(
            widths, heights, angles, units='xy', offsets=mu, linewidths=4,
            transOffset=axis.transData, facecolors='none', edgecolors=outline))
    if filled:
        axis.add_collection(EllipseCollection(
            widths, heights, angles, units='xy', offsets=mu, linewidths=2,
            transOffset=axis.transData, facecolors=fc, edgecolors=ec))
    else:
        axis.add_collection(EllipseCollection(
            widths, heights, angles, units='xy', offsets=mu, linewidths=2.5,
            transOffset=axis.transData, facecolors='none', edgecolors=color))


def GMM_pairplot(data, w, mu, C, limits=None, entropy=False):
    """Display 2D projections of a Gaussian mixture model fit.

    Parameters
    ----------
    data : pandas DataFrame
        N samples of D-dimensional data.
    w : array
        1D array of K relative weights for each ellipse. Must sum to one.
    mu : array
        Array of shape (K, 2) giving the 2-dimensional centroids of
        each ellipse.
    C : array
        Array of shape (K, 2, 2) giving the 2 x 2 covariance matrix for
        each ellipse.
    limits : array or None
        Array of shape (D, 2) giving [lo,hi] plot limits for each of the
        D dimensions. Limits are determined by the data scatter when None.
    """
    colnames = data.columns.values
    X = data.values
    N, D = X.shape
    if entropy:
        n_components = len(w)
        # Pick good colors to distinguish the different clusters.
        cmap = ListedColormap(
            sns.color_palette('husl', n_components).as_hex())
        # Calculate the relative probability that each sample belongs to each cluster.
        # This is equivalent to fit.predict_proba(X)
        lnprob = np.zeros((n_components, N))
        for k in range(n_components):
            lnprob[k] = scipy.stats.multivariate_normal.logpdf(X, mu[k], C[k])
        lnprob += np.log(w)[:, np.newaxis]
        prob = np.exp(lnprob)
        prob /= prob.sum(axis=0)
        prob = prob.T
        # Assign each sample to its most probable cluster.
        labels = np.argmax(prob, axis=1)
        color = cmap(labels)
        if n_components > 1:
            # Calculate the relative entropy (0-1) as a measure of cluster assignment ambiguity.
            relative_entropy = -np.sum(prob * np.log(prob), axis=1) / np.log(n_components)
            color[:, :3] *= (1 - relative_entropy).reshape(-1, 1)        
    # Build a pairplot of the results.
    fs = 5 * min(D - 1, 3)
    fig, axes = plt.subplots(D - 1, D - 1, sharex='col', sharey='row',
                             squeeze=False, figsize=(fs, fs))
    for i in range(1, D):
        for j in range(D - 1):
            ax = axes[i - 1, j]
            if j >= i:
                ax.axis('off')
                continue
            # Plot the data in this projection.
            if entropy:
                ax.scatter(X[:, j], X[:, i], s=5, c=color, cmap=cmap)
                draw_ellipses(
                    w, mu[:, [j, i]], C[:, [[j], [i]], [[j, i]]],
                    color='w', outline='k', filled=False, axis=ax)
            else:
                ax.scatter(X[:, j], X[:, i], s=10, alpha=0.3, c='k', lw=0)
                draw_ellipses(
                    w, mu[:, [j, i]], C[:, [[j], [i]], [[j, i]]],
                    color='red', outline=None, filled=True, axis=ax)
            # Overlay the fit components in this projection.
            # Add axis labels and optional limits.
            if j == 0:
                ax.set_ylabel(colnames[i])
                if limits: ax.set_ylim(limits[i])
            if i == D - 1:
                ax.set_xlabel(colnames[j])
                if limits: ax.set_xlim(limits[j])
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
