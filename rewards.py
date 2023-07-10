import numpy as np
import networkx as nx
from utils import (
    laplacian,
    signless_laplacian,
    nb_components,
    min_square,
    make_graph,
    make_matrix,
    EPSILON,
)
import math


INF = float("inf")


def get_reward_square_eigenvalues(state: np.ndarray[int], n: int) -> float:
    g = make_graph(state, n)
    if nx.number_connected_components(g) > 1:
        return -INF
    return n - 1 - min_square(g)


def get_reward_avg_deg(state: np.ndarray[int], n: int) -> float:
    return -abs(3 - np.sum(state) * 2 / n)


def get_reward_tree(state: np.ndarray[int], n: int) -> float:
    m = make_matrix(state, n)
    if nb_components(m) > 1:
        return -100
    return -sum(state)


def get_reward_deg(state: np.ndarray[int], n: int) -> float:
    g = make_matrix(state, n)
    return -abs(g.sum(axis=-1) - 3).sum()


def get_reward_brouwer(state: np.ndarray[int], n: int) -> float:
    t = 10
    g = make_matrix(state, n)
    l = laplacian(g)
    eigen_values = np.sort(np.linalg.eigvals(l))
    return sum(eigen_values[-t:]) - np.count_nonzero(state) - t * (t + 1) / 2


def get_reward_ashraf(state: np.ndarray[int], n: int) -> float:
    t = 10
    g = make_matrix(state, n)
    l = signless_laplacian(g)
    eigen_values = np.sort(np.linalg.eigvals(l))
    return sum(eigen_values[-t:]) - np.count_nonzero(state) - t * (t + 1) / 2


def get_reward_conj21(state: np.ndarray[int], n: int) -> float:
    """
    Calculates the reward for a given word.
    This function is very slow, it can be massively sped up with numba -- but numba doesn't support networkx yet,
    which is very convenient to use here

    :param state: the first MYN letters of this param are the word that the neural network has constructed.

    :returns: the reward (a real number). Higher is better, the network will try to maximize this.
    """

    # Example reward function, for Conjecture 2.1
    # Given a graph, it minimizes lambda_1 + mu.
    # Takes a few hours  (between 300 and 10000 iterations) to converge (loss < 0.01) on my computer with these parameters if not using parallelization.
    # There is a lot of run-to-run variance.
    # Finds the counterexample some 30% (?) of the time with these parameters, but you can run several instances in parallel.

    # Construct the graph
    g = make_graph(state, n)

    # g is assumed to be connected in the conjecture. If it isn't, return a very negative reward.
    if not (nx.is_connected(g)):
        return -INF

    # Calculate the eigenvalues of g
    eigen_vals = np.linalg.eigvalsh(nx.adjacency_matrix(g).todense())
    eigen_vals_real_abs = np.zeros_like(eigen_vals)
    for i in range(len(eigen_vals)):
        eigen_vals_real_abs[i] = abs(eigen_vals[i])
    lambda1 = max(eigen_vals_real_abs)

    # Calculate the matching number of g
    max_match = nx.max_weight_matching(g)
    mu = len(max_match)

    # Calculate the reward. Since we want to minimize lambda_1 + mu, we return the negative of this.
    # We add to this the conjectured best value. This way if the reward is positive we know we have a counterexample.
    reward = math.sqrt(n - 1) + 1 - lambda1 - mu

    return reward


def get_reward_special_case_conjecture_Aouchiche_Hansen_graph_energy(
    state: np.ndarray[int], n: int
) -> float:
    g = make_matrix(state, n)
    eigen_values = np.linalg.eigvals(g)
    strictly_positive_eigenvalues = [x for x in eigen_values if x > EPSILON]
    strictly_negative_eigenvalues = [x for x in eigen_values if x < -EPSILON]

    return -sum(strictly_positive_eigenvalues) + max(
        len(strictly_positive_eigenvalues), len(strictly_negative_eigenvalues)
    )


def get_reward_third_eigenvalue(state: np.ndarray[int], n: int) -> float:
    g = make_matrix(state, n)
    eigen_values = sorted(np.linalg.eigvals(g))
    return eigen_values[-4] - n/4
