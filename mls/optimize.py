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
    plt.grid('off')
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
