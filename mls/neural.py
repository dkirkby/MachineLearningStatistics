"""Neural network utilities for the UC Irvine course on
'ML & Statistics for Physicists'
"""
import numpy as np

import matplotlib.pyplot as plt


def nn_map2d(fx, params=None, x_range=3, ax=None, vlim=None, label=None):
    """
    """
    ax = ax or plt.gca()
    if vlim is None:
        vlim = np.max(np.abs(fx))
    ax.imshow(fx, interpolation='none', origin='lower', cmap='coolwarm',
              aspect='equal', extent=[-x_range, +x_range, -x_range, +x_range],
              vmin=-vlim, vmax=+vlim)
    if params:
        w, b = params
        w = np.asarray(w)
        x0 = -b * w / np.dot(w, w)
        ax.annotate('', xy=x0 + w, xytext=x0,
                    arrowprops=dict(arrowstyle='->', lw=3, color='k'))
    if label:
        ax.text(0.5, 0.9, label, horizontalalignment='center',
                color='k', fontsize=16, transform=ax.transAxes)
    ax.grid(False)
    ax.axis('off')


def nn_unit_draw2d(w, b, phi, x_range=3, nx=250, ax=None, vlim=None, label=None):
    """Draw a single network unit or layer with 2D input.

    Parameters
    ----------
    w : array
        1D array of weight values to use.
    b : float or array
        scalar (unit) or 1D array (layer) of bias values to use.
    phi : callable
        Activation function to use.
    """
    x_i = np.linspace(-x_range, +x_range, nx)
    X = np.stack(np.meshgrid(x_i, x_i), axis=2).reshape(nx ** 2, 2)
    W = np.asarray(w).reshape(2, 1)
    fx = phi(np.dot(X, W) + b).reshape(nx, nx)
    nn_map2d(fx, (w, b), x_range, ax, vlim, label)


def nn_graph_draw2d(*layers, x_range=3, label=None, n_grid=250, n_bins=25):
    """Draw the response of a neural network with 2D input.

    Parameters
    ----------
    *layers : tuple
        Each layer is specified as a tuple (W, b, phi).
    """
    x_i = np.linspace(-x_range, +x_range, n_grid)
    layer_in = np.stack(np.meshgrid(x_i, x_i), axis=0).reshape(2, -1)
    layer_out = [ ]
    layer_s = [ ]
    depth = len(layers)
    n_max, vlim = 0, 0.
    for i, (W, b, phi) in enumerate(layers):
        WT = np.asarray(W).T
        b = np.asarray(b)
        n_out, n_in = WT.shape
        n_max = max(n_max, n_out)
        if layer_in.shape != (n_in, n_grid ** 2):
            raise ValueError(
                'LYR{}: number of rows in W ({}) does not match layer input size ({}).'
                .format(i + 1, n_in, layer_in.shape))
        if b.shape != (n_out,):
            raise ValueError(
                'LYR{}: number of columns in W ({}) does not match size of b ({}).'
                .format(i + 1, n_out, b.shape))
        s = np.dot(WT, layer_in) + b.reshape(-1, 1)
        layer_s.append(s)
        layer_out.append(phi(s))
        layer_in = layer_out[-1]
        vlim = max(vlim, np.max(layer_in))
    _, ax = plt.subplots(n_max, depth, figsize=(3 * depth, 3 * n_max),
                         sharex=True, sharey=True, squeeze=False)
    for i, (W, b, phi) in enumerate(layers):
        W = np.asarray(W)
        for j in range(n_max):
            if j >= len(layer_out[i]):
                ax[j, i].axis('off')
                continue
            label = 'LYR{}-NODE{}'.format(i + 1, j + 1)
            params = (W[:,j], b[j]) if i == 0 else None
            fx = layer_out[i][j]
            nn_map2d(fx.reshape(n_grid, n_grid), params=params,
                      ax=ax[j, i], vlim=vlim, x_range=x_range, label=label)
            if i > 0 and n_bins:
                s = layer_s[i][j]
                s_min, s_max = np.min(s), np.max(s)
                t = 2 * x_range * (s - s_min) / (s_max - s_min) - x_range
                rhs = ax[j, i].twinx()
                hist, bins, _ = rhs.hist(
                    t, bins=n_bins, range=(-x_range, x_range),
                    histtype='stepfilled', color='w', lw=1, alpha=0.25)
                s_grid = np.linspace(s_min, s_max, 101)
                t_grid = np.linspace(-x_range, +x_range, 101)
                phi_grid = phi(s_grid)
                phi_min, phi_max = np.min(phi_grid), np.max(phi_grid)
                z_grid = (
                    (phi_grid - phi_min) / (phi_max - phi_min) * np.max(hist))
                rhs.plot(t_grid, z_grid, 'k--', lw=1, alpha=0.5)
                rhs.axvline(-x_range * (s_max + s_min) / (s_max - s_min),
                            c='w', lw=1, alpha=0.5)
                rhs.set_xlim(-x_range, +x_range)
                rhs.axis('off')
    plt.subplots_adjust(wspace=0.015, hspace=0.010,
                        left=0, right=1, bottom=0, top=1)
