"""Bayesian learning for the UC Irvine course on 'ML & Statistics for Physicists'
"""

import pandas as pd

import IPython.display


def Learn(prior, likelihood, *data):
    """Learn from data using Bayesian inference.

    Assumes that the model and data spaces are discrete.

    Parameters
    ----------
    prior : dict
        Dictionary of prior probabilities for all possible models.
    likelihood : callable
        Called with args (D,M) and must return a normalized likelihood.
    data : variable-length list
        Zero or more items of data to use in updating the prior.
    """
    # Initialize the Bayes' rule numerator for each model.
    prob = prior.copy()
    history = [('PRIOR', prior)]
    # Loop over data.
    for D in data:
        # Update the Bayes' rule numerator for each model.
        prob = {M: prob[M] * likelihood(D, M) for M in prob}
        # Calculate the Bayes' rule denominator.
        norm = sum(prob.values())
        # Calculate the posterior probabilities for each model.
        prob = {M: prob[M] / norm for M in prob}
        history.append(('D={}'.format(D), prob))
    # Print our learning history.
    index, rows = zip(*history)
    IPython.display.display(pd.DataFrame(list(rows), index=index).round(3))
