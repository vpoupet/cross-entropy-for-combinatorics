import numpy as np
import networkx as nx
import grinpy as gp
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


def get_reward_square_eigenvalues(state: np.ndarray, n: int) -> float:
    g = make_graph(state, n)
    if nx.number_connected_components(g) > 1:
        return -INF
    return n - 1 - min_square(g)


def get_reward_brouwer(state: np.ndarray, n: int) -> float:
    t = 10
    g = make_matrix(state, n)
    l = laplacian(g)
    eigen_values = np.linalg.eigvalsh(l)
    return sum(eigen_values[-t:]) - np.count_nonzero(state) - t * (t + 1) / 2


def get_reward_ashraf(state: np.ndarray, n: int) -> float:
    t = 10
    g = make_matrix(state, n)
    l = signless_laplacian(g)
    eigen_values = np.linalg.eigvalsh(l)
    return sum(eigen_values[-t:]) - np.count_nonzero(state) - t * (t + 1) / 2


def get_reward_conj21(state: np.ndarray, n: int) -> float:
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
    state: np.ndarray, n: int
) -> float:
    g = make_matrix(state, n)
    eigen_values = np.linalg.eigvalsh(g)
    strictly_positive_eigenvalues = [x for x in eigen_values if x > EPSILON]
    strictly_negative_eigenvalues = [x for x in eigen_values if x < -EPSILON]

    return -sum(strictly_positive_eigenvalues) + max(
        len(strictly_positive_eigenvalues), len(strictly_negative_eigenvalues)
    )


def get_reward_third_eigenvalue(state: np.ndarray, n: int) -> float:
    # see this : https://arxiv.org/pdf/2304.12324.pdf and https://arxiv.org/pdf/1502.00359.pdf
    g = make_matrix(state, n)
    eigen_values = np.linalg.eigvalsh(g)
    return eigen_values[-3] - n / 3


def get_reward_fourth_eigenvalue(state: np.ndarray, n: int) -> float:
    # see this : https://arxiv.org/pdf/2304.12324.pdf and https://arxiv.org/pdf/1502.00359.pdf
    g = make_matrix(state, n)
    eigen_values = np.linalg.eigvalsh(g)
    return eigen_values[-4] - 0.269*n


def get_reward_bollobas_nikiforov(state: np.ndarray, n: int) -> float:
    # see this : https://arxiv.org/pdf/2101.05229.pdf
    m = make_matrix(state, n)
    g = make_graph(state, n)
    eigen_values = np.linalg.eigvalsh(m)
    clique_number = gp.clique_number(g)
    return (
        -2.0 * np.sum(state) * (clique_number - 1) / clique_number
        + eigen_values[-1] * eigen_values[-1]
        + eigen_values[-2] * eigen_values[-2]
    )


def get_reward_Elphick_Linz_Wocjan(state: np.ndarray, n: int) -> float:
    # see this : https://arxiv.org/pdf/2101.05229.pdf
    m = make_matrix(state, n)
    g = make_graph(state, n)
    eigen_values = np.linalg.eigvalsh(m)
    strictly_positive_eigenvalues = [x for x in eigen_values if x > EPSILON]
    clique_number = gp.clique_number(g)
    l = min(len(strictly_positive_eigenvalues), clique_number)
    somme = 0
    for x in range(l):
        somme += strictly_positive_eigenvalues[x] * strictly_positive_eigenvalues[x]
    return -2.0 * np.sum(state) * (clique_number - 1) / clique_number + somme


def get_reward_Elphick_Wocjan(state: np.ndarray, n: int) -> float:
    # see this : https://arxiv.org/pdf/2101.05229.pdf
    m = make_matrix(state, n)
    g = make_graph(state, n)
    degree_sequence = [d for n, d in g.degree()]
    eigen_values = np.linalg.eigvalsh(m)
    strictly_positive_eigenvalues = [x for x in eigen_values if x > EPSILON]
    clique_number = gp.clique_number(g)
    if clique_number==0:
        return -INF
    somme = 0
    for x in range(len(strictly_positive_eigenvalues)):
        somme += strictly_positive_eigenvalues[x] * strictly_positive_eigenvalues[x]
    if np.sqrt(somme) - n*(1 - 1/clique_number) > EPSILON:
        print("Bingo elphick wocjan !")
        print(m)
    return  np.sqrt(somme) - n*(1 - 1/clique_number)


def get_reward_clique(state: np.ndarray, n: int) -> float:
    """
    Dummy reward function that aims to build a clique (more edges give better reward).

    Used only for testing purposes.
    """
    return 1 + sum(state) - (n * (n - 1) // 2)

def get_reward_randic_radius(state: np.ndarray, n: int) -> float:
    """
    Conjecture saying that Randic index can be lower bounded in terms of the graph radius.
    """
    import grinpy as gp
    g = make_graph(state,n)
    if not nx.is_connected(g):
        return -INF;
    randic_index = gp.randic_index(g)
    radius = nx.radius(g)
    if radius - randic_index > EPSILON:
        print("bingo randic !")
        print(state)
        print(make_graph(state,n))
        exit(1)
    return radius - randic_index

def get_reward_difference_szeged_wiener(state: np.ndarray, n: int) -> float:
    """
    Difference between Szeged and Wiener index.
    """
    g = make_graph(state,n)
    for i in range(n-1):
        g.add_edge(i,i+1)
    g.add_edge(n-1,0)

    # if not nx.is_biconnected(g):
    #     return -INF
    degree_sequence = [d for n, d in g.degree()]
    if all(x==n-1 for x in degree_sequence):
        return -INF
    distance_matrix = nx.floyd_warshall_numpy(g)
    wiener_index = np.sum(distance_matrix)/2
    szeged_index = 0;
    for e in g.edges():
        n0 = 0; n1 = 0;
        for i in range(n):
            if distance_matrix[e[0]][i] > distance_matrix[e[1]][i]:
                n0 += 1
            elif distance_matrix[e[0]][i] < distance_matrix[e[1]][i]:
                n1 += 1
        szeged_index += n0 * n1;
    if wiener_index == INF:
        return -INF
    if -szeged_index + wiener_index + 2*n > EPSILON:
        sorted_degree_sequence = np.sort(degree_sequence)
        if sorted_degree_sequence[0]==2 and sorted_degree_sequence[-1] == n-1 and sorted_degree_sequence[-2] == n-1:
            found = False
            for d in range(1,len(sorted_degree_sequence)-2):
                if sorted_degree_sequence[d] != n-2:
                    found = True; break
            if found:
                print("BINGO 1")
                print(state)
                print(make_matrix(state,n))
                exit(1)
            else:
                return -INF
        elif sorted_degree_sequence[0]==n-3 and sorted_degree_sequence[1] == n-2 and sorted_degree_sequence[2] == n-2:
            found = False
            for d in range(1,len(sorted_degree_sequence)-2):
                if sorted_degree_sequence[d] != n-1:
                    found = True; break
            if found:
                print("BINGO 2")
                print(state)
                print(make_matrix(state,n))
                exit(1)
            else:
                return -INF
        else:
            print("BINGO 3")
            print(make_matrix(state,n))
            exit(1)
    return -szeged_index + wiener_index + 2*n;


def get_reward_akbari_hosseinzadeh(state: np.ndarray, n: int) -> float:
    # see this : https://arxiv.org/pdf/2304.12324.pdf and https://arxiv.org/pdf/1502.00359.pdf
    g = make_graph(state, n)
    m = make_matrix(state, n)

    maxDegree = 0
    minDegree = n
    for i,d in g.degree():
        if d > maxDegree:
            maxDegree = d
        if d < minDegree:
            minDegree = d
    if minDegree == n-1 or n >= minDegree+maxDegree or 2*sum(state)+n*(n-1) >= (minDegree+maxDegree)^2:
        return -1000

    determinant = abs(np.linalg.det(m))
    eigen_vals = np.linalg.eigvalsh(m)
    if determinant <= EPSILON or abs(determinant) >= eigen_vals[-1]:
        return -1000
    somme = sum(abs(eigen_vals))
    if minDegree + maxDegree - somme > EPSILON:
        print(m)
    return minDegree + maxDegree - somme


mapping = {
    "square": get_reward_square_eigenvalues,
    "brouwer": get_reward_brouwer,
    "ashraf": get_reward_ashraf,
    "conj21": get_reward_conj21,
    "aouchiche": get_reward_special_case_conjecture_Aouchiche_Hansen_graph_energy,
    "third_ev": get_reward_third_eigenvalue,
    "fourth_ev": get_reward_fourth_eigenvalue,
    "bollobas": get_reward_bollobas_nikiforov,
    "elphick_linz_wocjan": get_reward_Elphick_Linz_Wocjan,
    "elphick_wocjan": get_reward_Elphick_Wocjan,
    "clique": get_reward_clique,
    "szeged_wiener": get_reward_difference_szeged_wiener,
    "akbari": get_reward_akbari_hosseinzadeh,
    "randic": get_reward_randic_radius
}
