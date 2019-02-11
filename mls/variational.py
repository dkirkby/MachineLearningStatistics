"""Variational inference utilities
for the UC Irvine course on 'ML & Statistics for Physicists'
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats


def calculate_KL(log_q, log_p, theta):
    """Calculate the KL divergence of q wrt p for single-parameter PDFs.

    Uses the trapezoid rule for the numerical integration. Integrals are only
    calculated over the input theta range, so are not valid when p or q have
    significant mass outside this range.

    Regions where either PDF is zero are handled correctly, although an
    integrable singularity due to p=0 will result in a divergent KL because the
    inputs are tabulated.

    Parameters
    ----------
    log_q : array
        Values of log q(theta, s) tabulated on a grid with shape (ns, ntheta)
        of s (axis=0) and theta (axis=1).
    log_p : array
        Values of log p(theta) tabulated on a grid with shape (ntheta) of theta.
    theta : array
        Values of theta where log_q and log_p are tabulated.

    Returns
    -------
    tuple
        Tuple (KL, integrand) where KL is an array of ns divergence values and
        integrand is an array with shape (ns, ntheta) of KL integrands.
    """
    # special handling for q=0.
    q_log_q = np.zeros_like(log_q)
    nonzero = log_q > -np.inf
    q_log_q[nonzero] = log_q[nonzero] * np.exp(log_q[nonzero])
    integrand = q_log_q - log_p * np.exp(log_q)
    return np.trapz(integrand, theta), integrand


def plot_KL(q, q_scale_range, p, p_scale, theta_range):
    """Explanatory plots for the KL divergence.

    q and p are arbitrary PDFs defined in scipy.stats. q represents a
    family of PDFs by allowing its scale factor to vary in some range.
    The target pdf p uses a fixed scale factor.

    Parameters
    ----------
    q : str
        Name of a 1D continous random variable defined in scipy.stats.
    q_scale_range : list
        List [lo, hi] giving the range of scale factors to allow in defining the
        q family of PDFs.
    p : str
        Name of a 1D continous random variable defined in scipy.stats.
    p_scale : float
        Fixed scale factor to use for the target PDF p.
    theta_range : list
        List [lo, hi] giving the range to use for plotting and integration.
    """
    q = getattr(scipy.stats, q)
    p = getattr(scipy.stats, p)

    theta = np.linspace(*theta_range, 251)
    log_p = p.logpdf(theta, scale=p_scale)
    assert np.all(np.isfinite(log_p))

    q_scale = np.linspace(*q_scale_range, 101)
    log_q = q.logpdf(theta, scale=q_scale[:, np.newaxis])

    KLs, KL_ints = calculate_KL(log_q, log_p, theta)
    ibest = np.argmin(KLs)

    fig = plt.figure(figsize=(12, 7))
    ax = [plt.subplot2grid((2,2), (0,0)), plt.subplot2grid((2,2), (1,0)),
          plt.subplot2grid((2,2), (0,1), rowspan=2)]
    cmap = sns.color_palette('bright', n_colors=1 + len(KLs)).as_hex()

    ax[0].plot(theta, np.exp(log_p), '-', lw=10, c=cmap[0],
               alpha=0.25, label='$p(\\theta)$')
    ax[0].axhline(0., color='gray', lw=1)
    ax[1].axhline(0., color='gray', lw=1)
    ax[2].axhline(0., color='gray', lw=1)
    ax[2].plot(q_scale, KLs, 'k-', label='KL$(q(s) \parallel p)$')
    for i, idx in enumerate((0, ibest, -1)):
        c = cmap[i + 1]
        label = '$q(\\theta;s={:.2f})$'.format(q_scale[idx])
        ax[0].plot(theta, np.exp(log_q[idx]), '--', lw=2,
                   alpha=1, c=c, label=label)
        ax[1].plot(theta, KL_ints[idx], '--', lw=2, alpha=1, c=c)
        ax[2].scatter(q_scale[idx], KLs[idx], lw=0, c=cmap[i + 1], s=150)
    ax[0].legend()
    ax[0].set_ylabel('$p(x), q(\\theta; s)$', fontsize='x-large')
    ax[0].set_xlim(*theta_range)
    ax[0].set_xticklabels([])
    ax[0].set_yticks([])
    ax[1].set_ylabel('KL$(q \parallel p)$ integrand', fontsize='x-large')
    ax[1].set_xlim(*theta_range)
    ax[1].set_xlabel('$\\theta$', fontsize='large')
    ax[1].set_yticks([])
    ax[2].set_xlabel('$q(\\theta;s)$ scale $s$', fontsize='large')
    ax[2].legend(loc='upper center', fontsize='x-large')
    plt.subplots_adjust(left=0.05, hspace=0.05, wspace=0.1)


def calculate_ELBO(log_q, log_likelihood, log_prior, theta):
    """Calculate the ELBO of q for single-parameter PDFs.
    """
    KLqP, integrand = calculate_KL(log_q, log_prior, theta)
    integrand = np.exp(log_q) * log_likelihood - integrand
    return np.trapz(integrand, theta), integrand


def plot_ELBO(q, q_scale_range, likelihood, prior, theta_range, n_data, seed=123):
    """Explanatory plots for the evidence lower bound (ELBO).

    Data is modeled with a single offset (loc) parameter theta with an arbitrary
    likelihood and prior. A random sample of generated data is used to calculate
    the posterior, which is approximated by adjusting the scale parameter of
    the arbitrary PDF family q.

    Parameters
    ----------
    q : str
        Name of a 1D continous random variable defined in scipy.stats.
    q_scale_range : list
        List [lo, hi] giving the range of scale factors to allow in defining the
        q family of PDFs.
    likelihood : str
        Name of a 1D continous random variable defined in scipy.stats.
    prior : str
        Name of a 1D continous random variable defined in scipy.stats.
    theta_range : list
        List [lo, hi] giving the range to use for plotting and integration.
        The true value of theta used to generate data is (lo + hi) / 2.
    n_data : int
        Number of data points to generate by sampling from the likelihood with
        theta = theta_true.
    seed : int
        Random number seed to use for reproducible results.
    """
    q = getattr(scipy.stats, q)
    likelihood = getattr(scipy.stats, likelihood)
    prior = getattr(scipy.stats, prior)

    # Generate random data using the midpoint of the theta range as the
    # true value of theta for sampling the likelihood.
    theta = np.linspace(*theta_range, 251)
    theta_true = 0.5 * (theta[0] + theta[-1])
    D = likelihood.rvs(
        loc=theta_true, size=n_data,
        random_state=np.random.RandomState(seed=seed))

    # Calculate the likelihood and prior for each theta.
    log_L = likelihood.logpdf(D, loc=theta[:, np.newaxis]).sum(axis=1)
    log_P = prior.logpdf(theta)

    # Calculate the evidence and posterior.
    log_post = log_L + log_P
    log_evidence = np.log(np.trapz(np.exp(log_post), theta))
    log_post -= log_evidence
    assert np.all(np.isfinite(log_post))

    q_scale = np.linspace(*q_scale_range, 101)
    log_q = q.logpdf(theta, scale=q_scale[:, np.newaxis])

    KLs, KL_ints = calculate_KL(log_q, log_post, theta)
    ibest = np.argmin(KLs)

    ELBOs, ELBO_ints = calculate_ELBO(log_q, log_L, log_P, theta)

    fig = plt.figure(figsize=(12, 8))
    ax = [plt.subplot2grid((2,2), (0,0)), plt.subplot2grid((2,2), (1,0)),
          plt.subplot2grid((2,2), (0,1)), plt.subplot2grid((2,2), (1,1))]
    cmap = sns.color_palette('bright', n_colors=1 + len(KLs)).as_hex()

    ax[0].plot(theta, np.exp(log_post), '-', lw=10, c=cmap[0],
               alpha=0.25, label='$P(\\theta\mid D)$')
    ax[0].axhline(0., color='gray', lw=1)
    ax[1].axhline(0., color='gray', lw=1)
    ax[2].axhline(0., color='gray', lw=1)
    ax[2].plot(q_scale, KLs, 'k-', label='KL$(q(s) \parallel p)$')
    ax[2].plot(q_scale, log_evidence - ELBOs, 'k:', lw=6,
               alpha=0.5, label='$\log P(D) - ELBO(q(s))$')
    for i, idx in enumerate((0, ibest, -1)):
        c = cmap[i + 1]
        label = '$q(\\theta;s={:.2f})$'.format(q_scale[idx])
        ax[0].plot(theta, np.exp(log_q[idx]), '--', lw=2,
                   alpha=1, c=c, label=label)
        ax[1].plot(theta, KL_ints[idx], '--', lw=2, alpha=1, c=c)
        ax[2].scatter(q_scale[idx], KLs[idx], lw=0, c=c, s=150)
    ax[0].legend()
    ax[0].set_ylabel('$p(x), q(\\theta; s)$', fontsize='x-large')
    ax[0].set_xlim(*theta_range)
    ax[0].set_xlabel('Model parameter $\\theta$', fontsize='large')
    ax[0].set_yticks([])
    ax[1].set_ylabel('KL$(q \parallel p)$ integrand', fontsize='x-large')
    ax[1].set_xlim(*theta_range)
    ax[1].set_xlabel('Model parameter $\\theta$', fontsize='large')
    ax[1].set_yticks([])
    ax[2].set_xlabel('$q(\\theta;s)$ scale $s$', fontsize='large')
    ax[2].legend(loc='upper center', fontsize='x-large')

    x_lim = 1.1 * np.max(np.abs(D))
    ax[3].hist(D, density=True, range=(-x_lim, +x_lim), histtype='stepfilled')
    x = np.linspace(-x_lim, +x_lim, 250)
    dtheta = 0.25 * (theta[-1] - theta[0])
    for theta, ls in zip(
        (theta_true - dtheta, theta_true, theta_true + dtheta),
        ('--', '-', ':')):
        label = '$P(x\mid \\theta={:+.2f})$'.format(theta)
        ax[3].plot(x, likelihood.pdf(x, loc=theta), 'k', ls=ls, label=label)
    ax[3].set_xlabel('Observed sample $x$')
    ax[3].set_xlim(-x_lim, +x_lim)
    ax[3].legend()

    plt.subplots_adjust(
        left=0.05, right=0.95, hspace=0.25, wspace=0.15, top=0.95)
    fig.suptitle(
        '$\\theta_{\text{true}}' + ' = {:.2f}$ , $\log P(D) = {:.1f}$'
        .format(theta_true, log_evidence), fontsize='large')
