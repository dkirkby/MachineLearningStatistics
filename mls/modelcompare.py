"""Model comparison utilities for the UC Irvine course on
'ML & Statistics for Physicists'
"""
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import mixture


def cv_summary(cv):
    """Summarize the results from a GridSearchCV fit.

    Summarize a cross-validation grid search in a pandas DataFrame with the
    following transformations of the full results:
      - Remove all columns with timing measurements.
      - Remove the 'param_' prefix from column names.
      - Remove the '_score' suffix from column names.
      - Round scores to 3 decimal places.

     If the parameter grid is 1D, then this function also plots the test
     and training R2 scores versus the parameter.

    Parameters
    ----------
    cv : sklearn.model_selection.GridSearchCV
        Instance of a GridSearchCV object that has been fit to some data.

    Returns
    -------
    pandas.DataFrame
        Summary table of cross-validation results.
    """
    # Look up the list of parameters used in the grid.
    params = list(cv.cv_results_['params'][0].keys())
    # Index results by the test score rank.
    index = cv.cv_results_['rank_test_score']
    df = pd.DataFrame(cv.cv_results_, index=index).drop(columns=['params', 'rank_test_score'])
    # Remove columns that measure running time.
    df = df.drop(columns=[n for n in df.columns.values if n.endswith('_time')])
    # Remove param_ prefix from column names.
    df = df.rename(lambda n: n[6:] if n.startswith('param_') else n, axis='columns')
    # Remove _score suffix from column names.
    df = df.rename(lambda n: n[:-6] if n.endswith('_score') else n, axis='columns')
    if len(params) == 1:
        # Plot the test and training scores vs the grid parameter when there is only one.
        plt.plot(df[params[0]], df['mean_train'], 'o:', label='train')
        plt.plot(df[params[0]], df['mean_test'], 'o-', label='test')
        plt.legend(fontsize='x-large')
        plt.xlabel('Hyperparameter value')
        plt.ylabel('Score $R^2$')
        plt.ylim(max(-2, np.min(df['mean_test'])), 1)
    return df.sort_index().round(3)


def create_param_grid(samples, n_grid=50):
    """Create a parameter grid from parameter samples.

    Grids are based on 1D quantiles in each parameter, so are not uniform.

    Parameters
    ----------
    samples : array
        2D array with shape (N, P) containing N samples for P parameters.
    n_grid : int
        Number of grid points to use along each parameter axis.  The full
        grid contains n_grid ** P points.

    Returns
    -------
    array
        Array of shape (n_grid ** P, P) with parameters values covering the
        full grid.  Can be reshaped to ([P] * (P+1)) to reconstruct the
        P-dimensional grid structure.
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
    """Estimate the log evidence using MCMC samples.

    The evidence is estimated at each grid point using the ratio of the
    log_numerator and the empirical density of samples. Only grid points with
    the largest log_numerator are used for the estimate. The density is
    estimated with a Gaussian mixture model using the a number of components
    that minimizes the spread of estimates over the selected grid points.

    Parameters
    ----------
    samples : array
        2D array with shape (N, P) containing N samples for P parameters.
    param_grid : array
        2D array with shape (n_grid ** P, P) to specify a grid that covers
        the full parameter space. Normally obtained by calling
        :func:`create_param_grid`.
    log_numerator : array
        1D array with shape (n_grid ** P,) with the log_likelihood+log_prior
        value tabulated on the input parameter grid.
    max_components : int
        Maximum number of Gaussian mixture model components to use in
        estimating the density of samples over the parameter space.
    grid_fraction : float
        The fraction of grid points with the highest log_numerator to use
        for estimating the log_evidence.
    plot : bool
        When True, draw a scatter plot of log_numerator vs log_evidence for the
        requested fraction of grid points, using the best found number of
        GMM components.
    seed : int or None
        Random seed to use for reproducible results.

    Returns
    -------
    float
        Estimate of the log evidence.
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
