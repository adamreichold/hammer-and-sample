import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import norm
from scipy.special import expit

from hammer_and_sample import sample


data = pd.read_csv("data.csv")
params = pd.read_csv("params.csv")

groups = []
outcomes = []

for group, data in data.groupby("group", sort=False):
    groups.append(group)
    outcomes.append(np.bincount(data["outcome"]))


def log_prob(state):
    logit_theta = state[0]
    sigma = state[1]

    # prior on logit_theta
    log_prob = norm.logpdf(logit_theta, -2, 2)

    # prior on sigma (half normal)
    if sigma < 0:
        return float("-inf")

    log_prob += norm.logpdf(sigma, 0, 2)

    for group in range(len(outcomes)):
        group_alpha = state[2 + group]
        group_theta = expit(logit_theta + group_alpha)

        # likelihood of group_alpha given sigma
        log_prob += norm.logpdf(group_alpha, 0, sigma)

        # likelihood of data given group_theta
        failures, successes = outcomes[group]

        log_prob += successes * np.log(group_theta) + failures * np.log(1 - group_theta)

    return log_prob


rng = default_rng()


def guess_params():
    guess_logit_theta = rng.normal(-2, 2)
    guess_sigma = abs(rng.normal(0, 2))

    guess_alpha = [rng.normal(0, guess_sigma) for group in range(len(outcomes))]

    return np.array([guess_logit_theta, guess_sigma] + guess_alpha)


guess = [guess_params() for _ in range(100)]

chain, accepted = sample(log_prob, guess, 0, 1000)

converged_chain = chain[100 * 250 :]

estimated_logit_theta = np.sum(converged_chain[:, 0]) / len(converged_chain)

for group in range(len(outcomes)):
    true_group_theta = params[params["group"] == groups[group]].iloc[0]["theta"]

    estimated_alpha = np.sum(converged_chain[:, 2 + group]) / len(converged_chain)

    estimated_group_theta = expit(estimated_logit_theta + estimated_alpha)

    assert abs(true_group_theta - estimated_group_theta) < 0.01

acceptance_rate = accepted / len(chain)

assert acceptance_rate > 0.3 and acceptance_rate < 0.4
