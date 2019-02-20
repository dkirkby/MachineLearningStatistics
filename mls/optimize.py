"""Optimization utilities
for the UC Irvine course on 'ML & Statistics for Physicists'
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_rosenbrock(xrange=(-1.5, 1.5), yrange=(-0.5,1.5), ngrid=500,
                    shaded=True, path=None, all_calls=None):
    """Plot the Rosenbrock function with some optional decorations.

    Parameters
    ----------
    xrange : tuple
        Tuple (xlo, xhi) with the x range to plot.
    yrange : tuple
        Tuple (ylo, yhi) with the y range to plot.
    ngrid : int
        Number of grid points along x and y to use.
    shaded : bool
        Draw a shaded background using a log scale for the colormap.
    path : array or None
        Array of shape (npath,2) with (x,y) coordinates along a path to draw.
        A large "X" will mark the starting point.
    all_calls : array or None
        Array of shape (ncall,2) with (x,y) coordinates of additional calls to
        indicate on the plot. Only points not in path will be visually distinct.
    """
    # Tabulate the Rosenbrock function on the specified grid.
    x_grid = np.linspace(*xrange, ngrid)
    y_grid = np.linspace(*yrange, ngrid)
    f = (1 - x_grid) ** 2 + 100.0 * (y_grid[:, np.newaxis] - x_grid ** 2) ** 2
    # Plot the function.
    fig = plt.figure(figsize=(9, 5))
    if shaded:
        plt.imshow(np.log10(f), origin='lower', extent=[*xrange, *yrange],
                   cmap='plasma', aspect='auto')
    c =plt.contour(x_grid, y_grid, f, levels=[0.1, 1., 10., 100.],
                   colors='w', linewidths=1, linestyles='-')
    plt.clabel(c, inline=1, fontsize=10, fmt='%.0g')
    plt.axhline(1, c='gray', lw=1, ls='--')
    plt.axvline(1, c='gray', lw=1, ls='--')
    plt.grid(False)
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    if all_calls:
        plt.scatter(*np.array(all_calls).T, lw=0, s=10, c='cyan')
    if path:
        path = np.array(path)
        plt.scatter(*path.T, lw=0, s=10, c='k')
        plt.plot(*path.T, lw=1, c='k', alpha=0.3)
        plt.scatter(*path[0], marker='x', s=250, c='b')
    plt.xlim(*xrange)
    plt.ylim(*yrange)
    return xrange, yrange


def plot_posterior(D, mu_range=(-0.5,0.5), sigma_range=(0.7,1.5), ngrid=100,
                   path=None, VI=None, MC=None):
    """Plot a posterior with optional algorithm results superimposed.

    Assumes a Gaussian likelihood with parameters (mu, sigma) and flat
    priors in mu and t=log(sigma).

    Parameters
    ----------
    D : array
        Dataset to use for the true posterior.
    mu_range : tuple
        Limits (lo, hi) of mu to plot
    sigma_range : tuple
        Limits (lo, hi) of sigma to plot.
    ngrid : int
        Number of grid points to use for tabulating the true posterior.
    path : array or None
        An array of shape (npath, 2) giving the path used to find the MAP.
    VI : array or None
        Values of the variational parameters (s0, s1, s2, s3) to use to
        display the closest variational approximation.
    MC : tuple
        Tuple (mu, sigma) of 1D arrays with the same length, consisting of
        MCMC samples of the posterior to display.
    """
    # Create a grid covering the (mu, sigma) parameter space.
    mu = np.linspace(*mu_range, ngrid)
    sigma = np.linspace(*sigma_range, ngrid)
    sigma_ = sigma[:, np.newaxis]
    log_sigma_ = np.log(sigma_)

    # Calculate the true -log(posterior) up to a constant.
    NLL = np.sum(0.5 * (D[:, np.newaxis, np.newaxis] - mu) ** 2 / sigma_** 2 +
                 log_sigma_, axis=0)
    # Apply uniform priors on mu and log(sigma)
    NLP = NLL - log_sigma_
    NLP -= np.min(NLP)

    if VI is not None:
        s0, s1, s2, s3 = VI
        # Calculate the VI approximate -log(posterior) up to a constant.
        NLQ = (0.5 * (mu - s0) ** 2 / np.exp(s1) ** 2 +
               0.5 * (log_sigma_ - s2) ** 2 / np.exp(s3) ** 2)
        NLQ -= np.min(NLQ)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(NLP, origin='lower', extent=[*mu_range, *sigma_range],
               cmap='viridis_r', aspect='auto', vmax=16)
    c = plt.contour(mu, sigma, NLP, levels=[1, 2, 4, 8],
                    colors='w', linewidths=1, linestyles='-')
    plt.clabel(c, inline=1, fontsize=10, fmt='%.0g')
    plt.plot([], [], 'w-', label='true posterior')
    if path is not None:
        plt.scatter(*np.array(path).T, lw=0, s=15, c='r')
        plt.plot(*np.array(path).T, lw=0.5, c='r', label='MAP optimization')
        plt.scatter(*path[0], marker='x', s=250, c='r')
    if VI is not None:
        plt.contour(mu, sigma, NLQ, levels=[1, 2, 4, 8],
                    colors='r', linewidths=2, linestyles='--')
        plt.plot([], [], 'r--', label='VI approximation')
    if MC is not None:
        mu, sigma = MC
        plt.scatter(mu, sigma, s=15, alpha=0.8, zorder=10, lw=0, c='r',
                    label='MC samples')
    l = plt.legend(ncol=3, loc='upper center')
    plt.setp(l.get_texts(), color='w', fontsize='x-large')
    plt.axhline(1, c='gray', lw=1, ls='--')
    plt.axvline(0, c='gray', lw=1, ls='--')
    plt.grid(False)
    plt.xlabel('Offset parameter $\mu$')
    plt.ylabel('Scale parameter $\sigma$')
    plt.xlim(*mu_range)
    plt.ylim(*sigma_range)
