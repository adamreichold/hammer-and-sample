import numpy as np
from numpy.random import default_rng

from hammer_and_sample import sample


true_p = 0.75

rng = default_rng(seed=0)

data = rng.random(size=1000) < true_p

outcomes = np.bincount(data)


def log_prob(state):
    p = state[0]

    if p < 0 or p > 1:
        return float("-inf")

    return outcomes[0] * np.log(1 - p) + outcomes[1] * np.log(p)


guess = [rng.random(size=1) for _ in range(100)]

chain, accepted = sample(log_prob, guess, 0, 1000)

converged_chain = chain[100 * 100 :]

estimated_p = np.sum(converged_chain) / len(converged_chain)

assert abs(true_p - estimated_p) < 0.05

acceptance_rate = accepted / len(chain)

assert acceptance_rate > 0.7 and acceptance_rate < 0.8
