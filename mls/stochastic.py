"""Stochastic process utilities
for the UC Irvine course on 'ML & Statistics for Physicists'
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


class StochasticProcess(object):
    """Base class for a stochastic process.

    A subclass must implement the :meth:`initial` and :meth:`update` methods.
    """
    def __init__(self, seed=123):
        """Initialize a generic stochastic process.

        Parameters
        ----------
        seed : int or None
            Random seed to use for reproducible random numbers. A random state
            initialized with this seed is passed to the initial() and update()
            methods.
        """
        self.gen = np.random.RandomState(seed=seed)

    def plot(self, nsamples_per_run=50, nruns=3, joined=True):
        """Plot a few sequences of many samples.

        Parameters
        ----------
        nsamples_per_run : int
            Number of samples to plot for each run of the process.
        nruns : int
            Number of independent runs to plot. Should usually be a small
            number.
        joined : bool
            Join samples from the same run with lines when True.
        """
        cmap = sns.color_palette().as_hex()
        for i in range(nruns):
            run = self.run(nsamples_per_run)
            plt.plot(run, '.', c=cmap[i % len(cmap)])
            if joined:
                plt.plot(run, '-', alpha=0.2, c=cmap[i])
        plt.xlabel('Sequence number $n$')
        plt.ylabel('Value $x_n$')

    def pairplot(self, nsamples_per_run=4, nruns=500, x0cut=None):
        """Plot 1D and 2D statistics of a few samples using many runs.

        Uses a seaborn PairGrid.

        Parameters
        ----------
        nsamples_per_run : int
            Number of samples to include in the plot. Should usually be
            a small number.
        nruns : int
            Number of independent runs to use for building up statistics.
        x0cut : float or None
            Each plot is color-coded according to whether x0 is below or
            above this cut value, in order to show how dependencies propagate
            to later samples. Uses the median x0 value when None.
        """
        X = np.empty((nruns, nsamples_per_run))
        for i in range(nruns):
            X[i] = self.run(nsamples_per_run)
        names = ('$x_{{{}}}$'.format(j) for j in range(nsamples_per_run))
        df = pd.DataFrame(X, columns=names)
        # Color samples based on whether x0 > x0cut.
        x0 = X[:, 0]
        if x0cut is None:
            x0cut = np.median(x0)
        df['sel0'] = pd.cut(x0, [np.min(x0), x0cut, np.max(x0)])
        grid = sns.PairGrid(df, hue='sel0')
        grid.map_diag(plt.hist, histtype='stepfilled', alpha=0.4, lw=0)
        grid.map_diag(plt.hist, histtype='step', lw=2)
        grid.map_lower(plt.scatter, edgecolor='w', lw=0.5, s=20)

    def tabulate_conditional(self, n, m, lo, hi, nbins, nruns):
        """Tabulate the conditional probability P(Xm|Xn) numerically.

        n : int
            Tabulated probabilities are conditioned on n >= 0.
        m : int
            Tabulated probabilities are for P(Xm|Xn) with m > m.
        lo : float
            Tabulate values of Xn and Xm on the interval [lo, hi].
        hi : float
            Tabulate values of Xn and Xm on the interval [lo, hi].
        nbins : int
            Number of bins to use for tabulated values in [lo, hi].
        nruns : int
            Number of independent runs to perform to tabulate statistics.

        Returns
        -------
        tuple
            Tuple (bins, P) where bins is an array of nbins+1 bin edge values
            spanning [lo, hi] and P is an array of shape (nbins, nbins)
            containing the tabulated probabilities.  P is normalized for
            each value of the conditional Xn, i.e., P.sum(axis=1) = 1.
        """
        assert m > n and n >= 0
        nsteps = m - n
        result = np.empty((nbins, nbins))
        bins = np.linspace(lo, hi, nbins + 1)
        centers = 0.5 * (bins[1:] + bins[:-1])
        for i, Xi in enumerate(centers):
            Xj = []
            for j in range(nruns):
                history = [Xi]
                for k in range(nsteps):
                    history.append(self.update(history, self.gen))
                Xj.append(history[-1])
            result[i], _ = np.histogram(Xj, bins, normed=True)
        result *= (hi - lo) / nbins
        assert np.allclose(result.sum(axis=1), 1)
        return bins, result

    def plot_conditional(self, bins, table, xlabel=None, ylabel=None,
                         show_mean=True, ax=None):
        """Plot a single tabulated conditional probability P(Xm|Xn).

        Parameters
        ----------
        bins : numpy array
            An array of nbins+1 bin edge values where conditional
            probabilities are tabulated in table. Usually obtained using
            :meth:`tabulate_conditional`.
        table : numy array
            An array of tabulated conditional probalities.
            Usually obtained using :meth:`tabulate_conditional`.
        xlabel : str or None
            Label to use for the variable Xm in P(Xm|Xn).
        ylabel : str or None
            Label to use for the variable Xn in P(Xm|Xn).
        show_mean : bool
            Calculate and plot the mean <Xm> under P(Xm|Xn) for each Xn.
        ax : matplotlib axis or None
            Use the specified axes for drawing or the current axes.
        """
        lo, hi = bins[0], bins[-1]
        if ax is None:
            ax = plt.gca()
        ax.imshow(table, interpolation='none', origin='lower',
                  extent=[lo, hi, lo, hi])
        if show_mean:
            xy = 0.5 * (bins[1:] + bins[:-1])
            mean = np.sum(xy * table, axis=1) / np.sum(table, axis=1)
            ax.plot(mean, xy , 'b-')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid('off')

    def plot_conditionals(self, lo=0., hi=1., nbins=50, nruns=2000,
                          which=(1, 2, 3)):
        """Plot one or more sequential conditional probabilities.

        The initial probability P(X1|X0) is calculated using
        :meth:`tabulate_conditional` and each probability is plotted using
        :meth:`plot_conditional`. P(Xn|X0) is calculated as P(X1|X0) ** n.

        Parameters
        ----------
        lo : float
            Tabulate values of Xn and Xm on the interval [lo, hi].
        hi : float
            Tabulate values of Xn and Xm on the interval [lo, hi].
        nbins : int
            Number of bins to use for tabulated values in [lo, hi].
        nruns : int
            Number of independent runs to perform to tabulate statistics.
        which : iterable or ints
            Which conditional(s) to plot.
        """
        bins, T0 = self.tabulate_conditional(0, 1, lo, hi, nbins, nruns)
        T = T0.copy()
        if isinstance(which, int):
            which = (which,)
        n = len(which)
        fig, ax = plt.subplots(
            1, n, sharex=True, sharey=True, figsize=(4.2 * n, 4), squeeze=False)
        ylabel = '$X_0$'
        idx = 0
        for i in range(1, max(which) + 1):
            if i in which:
                xlabel = '$X_{{{}}}$'.format(i)
                self.plot_conditional(bins, T, xlabel, ylabel, ax=ax[0, idx])
                idx += 1
            T = T.dot(T0)
        plt.subplots_adjust(wspace=0.1)

    def run(self, nsamples_per_run):
        """Perform a single run of the stochastic process.

        Calls :meth:`initial` to get the initial value then calls
        :meth:`update` `nsamples_per_run-1` times to complete the run.

        Parameters
        ----------
        nsamples_per_run : int
            Number of samples to generate in this run, including the
            initial value.

        Returns
        -------
        numpy array
            1D array of generated values, of length `nsamples_per_run`.
        """
        history = [ self.initial(self.gen) ]
        for i in range(nsamples_per_run - 1):
            history.append(self.update(history, self.gen))
        return np.array(history)

    def initial(self, gen):
        """Return the initial value to use for a run.

        Parameters
        ----------
        gen : numpy.RandomState
            Use this object to generate any random numbers, for reproducibility.

        Returns
        -------
        float
            The initial value to use.
        """
        raise NotImplementedError

    def update(self, history, gen):
        """Return the next value to update a run.

        Parameters
        ----------
        history : list
            List of values generated so far.  Will always include at least
            one element (the initial value).
        gen : numpy.RandomState
            Use this object to generate any random numbers, for reproducibility.

        Returns
        -------
        float
            The next value to use.
        """
        raise NotImplementedError
