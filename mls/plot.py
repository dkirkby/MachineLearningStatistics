"""Plotting support for the UC Irvine course on 'ML & Statistics for Physicists'

Matplotlib must be installed to import this module.
"""

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import colorConverter
from matplotlib.collections import EllipseCollection


def draw_ellipses(w, mu, C, nsigmas=2, color='red', axis=None):
    """Draw a collection of ellipses.

    Uses the low-level EllipseCollection to efficiently draw a large number
    of ellipses. Useful to visualize the results of a GMM fit via
    GMM_pairplot() defined below.

    Parameters
    ----------
    w : array
        1D array of K relative weights for each ellipse. Must sum to one.
        Ellipses with smaller weights are rended with greater transparency.
    mu : array
        Array of shape (K, 2) giving the 2-dimensional centroids of
        each ellipse.
    C : array
        Array of shape (K, 2, 2) giving the 2 x 2 covariance matrix for
        each ellipse.
    axis : matplotlib axis or None
        Plot axis where the ellipse collection should be drawn. Uses the
        current default axis when None.
    """
    # Use transparency to indicate relative weights.
    color = colorConverter.to_rgba(color)
    ec = np.tile([color], (len(w), 1))
    ec[:, -1] *= w
    fc = np.tile([color], (len(w), 1))
    fc[:, -1] *= w ** 2
    # Calculate the ellipse angles and bounding boxes using SVD.
    U, s, _ = np.linalg.svd(C)
    angles = np.degrees(np.arctan2(U[:, 1, 0], U[:, 0, 0]))
    widths, heights = 2 * nsigmas * np.sqrt(s.T)
    # Data limits must already be defined for axis.transData to be valid.
    axis = axis or plt.gca()
    axis.add_collection(EllipseCollection(
        widths, heights, angles, units='xy', offsets=mu, linewidths=2,
        transOffset=axis.transData, facecolors=fc, edgecolors=ec))


def GMM_pairplot(data, w, mu, C, limits=None):
    """Display 2D projections of a Gaussian mixture model fit.

    Parameters
    ----------
    data : pandas DataFrame
        N samples of D-dimensional data.
    w : array
        1D array of K relative weights for each component. Must sum to one.
        Ellipses with smaller weights are rended with greater transparency.
    mu : array
        Array of shape (K, D) giving the D-dimensional centroids of
        each ellipse.
    C : array
        Array of shape (K, D, D) giving the D x D covariance matrix for
        each ellipse.
    limits : array or None
        Array of shape (D, 2) giving [lo,hi] plot limits for each of the
        D dimensions. Limits are determined by the data scatter when None.
    """
    colnames = data.columns.values
    X = data.values
    N, D = X.shape
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
            ax.scatter(X[:, j], X[:, i], s=10, alpha=0.3, c='k', lw=0)
            # Overlay the fit components in this projection.
            draw_ellipses(w, mu[:, [j, i]], C[:, [[j], [i]], [[j, i]]], axis=ax)
            # Add axis labels and optional limits.
            if j == 0:
                ax.set_ylabel(colnames[i])
                if limits: ax.set_ylim(limits[i])
            if i == D - 1:
                ax.set_xlabel(colnames[j])
                if limits: ax.set_xlim(limits[j])
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
