"""Model comparison utilities for the UC Irvine course on
'ML & Statistics for Physicists'
"""
import numpy as np

import matplotlib.pyplot as plt

from sklearn import mixture


def create_param_grid(samples, n_grid=50):
    """
    """
    samples = np.asarray(samples)
    N, P = samples.shape
    # Create a grid that is equally spaced in quantiles of each column.
    quantiles = np.linspace(0, 100, n_grid)
    grid = [np.percentile(column, quantiles) for column in samples.T]
    # Flatten to an array P-tuples that cover the grid.
    return np.moveaxis(np.stack(
        np.meshgrid(*grid), axis=-1), (0,1), (1,0)).reshape(-1, P)


def estimate_log_evidence(samples, param_grid, log_numerator, max_components=5,
                          grid_fraction=0.1, plot=True, seed=123):
    """
    """
    samples = np.asarray(samples)
    # Only use grid points with the highest log_numerator value.
    cut = np.percentile(log_numerator, 100. * (1 - grid_fraction))
    use = np.where(log_numerator >= cut)[0]
    use_grid = param_grid[use]
    use_log_numerator = log_numerator[use]
    # Loop over the number of GMM components.
    gen = np.random.RandomState(seed=seed)
    log_evidence = np.empty((max_components, len(use)))
    for i in range(max_components):
        # Fit the samples to a Gaussian mixture model.
        fit = mixture.GaussianMixture(
            n_components=i + 1, random_state=gen).fit(samples)
        # Evaluate the density on the grid.
        use_log_density = fit.score_samples(use_grid)
        # Estimate the log(evidence) from each grid point.
        log_evidence[i] = use_log_numerator - use_log_density
    # Calculate the median and 90% spread.
    lo, med, hi = np.percentile(log_evidence, (5, 50, 95), axis=-1)
    spread = 0.5 * (hi - lo)
    best = np.argmin(spread)
    if plot:
        plt.scatter(use_log_numerator, log_evidence[best], s=10, lw=0)
        plt.xlabel('$\log P(D\mid \Theta, M) + \log P(\Theta\mid M)$')
        plt.ylabel('$\log P(D\mid M)$')
        plt.axhline(lo[best], ls='--', c='k')
        plt.axhline(hi[best], ls='--', c='k')
        plt.axhline(med[best], ls='-', c='k')
        plt.title('n_GMM={}, logP(D|M)={:.3f}'.format(best + 1, med[best]))
        plt.show()
    return med[best]
