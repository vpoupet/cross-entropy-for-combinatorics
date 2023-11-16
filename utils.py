import functools
import math

import networkx as nx
import numpy as np

EPSILON = 1e-10
INF = float("inf")


def get_nb_vertices(nb_edges: int) -> int:
    return int((1 + np.sqrt(1 + 8 * nb_edges)) / 2)


def k(i, j, n):
    return n * i - (i * (i + 1)) // 2 + j - i - 1


def laplacian(m: np.ndarray[int]) -> np.ndarray[int]:
    """Returns the laplacian of a graph given by its adjacency matrix.

    The laplacian of a graph is defined as the difference between the degree matrix and the adjacency matrix.
    """
    return -m + np.diag(m.sum(axis=0))
    # l = -m
    # for i in range(len(m)):
    #     l[i, i] = m[i, :].sum()
    # return l


def signless_laplacian(m: np.ndarray[int]) -> np.ndarray[int]:
    """
    Returns the signless laplacian of a graph given by its adjacency matrix.

    The signless laplacian of a graph is defined as the sum of the degree matrix and the adjacency matrix.
    """
    return m + np.diag(m.sum(axis=0))
    # l = m[:, :]
    # for i in range(len(l)):
    #     l[i, i] = m[i, :].sum()
    # return l


def nb_components(m: np.ndarray[int]) -> int:
    """
    Returns the number of connected components of a graph given by its adjacency matrix.
    """
    l = laplacian(m)
    eigen_values = np.linalg.eigvals(l)
    return len(eigen_values) - np.count_nonzero(eigen_values)


def min_square(m: np.ndarray[int], epsilon: float = EPSILON) -> float:
    """
    Returns the min of the square of the sum of all positive eigenvalues and the square of the sum of all negative eigenvalues.
    """
    eigen_values = np.linalg.eigvals(m)
    s_plus = sum(e * e for e in eigen_values if e > epsilon)
    s_minus = sum(e * e for e in eigen_values if e < epsilon)
    return min(s_plus, s_minus)


def make_graph(state: np.ndarray[int], n: int) -> nx.Graph:
    """
    Returns a graph of n vertices from a state representation.

    The state representation of a graph with n vertices is an array of n(n-1)/2 bits. The first (n-1) bits encode
    the neighbors of vertex 0, the next (n-2) bits represent the neighbors of vertex 1 (other than 0) and so on.
    """
    g = nx.Graph()
    g.add_nodes_from(list(range(n)))
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if state[count] == 1:
                g.add_edge(i, j)
            count += 1
    return g


def make_matrix(state: np.ndarray[int], n: int) -> np.ndarray[int]:
    """
    Returns the adjacency matrix of a graph of n vertices from a state representation.

    The state representation of a graph with n vertices is an array of n(n-1)/2 bits. The first (n-1) bits encode
    the neighbors of vertex 0, the next (n-2) bits represent the neighbors of vertex 1 (other than 0) and so on.
    """

    m = np.zeros((n, n), dtype=int)
    k = 0
    for i in range(n - 1):
        m[i, i + 1 :] = state[k : k + n - i - 1]
        m[i + 1 :, i] = state[k : k + n - i - 1]
        k += n - i - 1
    return m


def make_matrix_from_string(s: str) -> np.ndarray[int]:
    return make_matrix([int(x) for x in s], get_nb_vertices(len(s)))


def improve_graph(state, n, reward_function):
    m = n * (n - 1) // 2
    best_reward = reward_function(state, n)
    best_state = state.copy()
    did_improve = False
    for i in range(m):
        state[i] = 1 - state[i]
        reward = reward_function(state, n)
        if reward > best_reward:
                best_reward = reward
                best_state = state.copy()
                did_improve = True
        state[i] = 1 - state[i]
    if did_improve:
        return improve_graph(best_state, n, reward_function)
    else:
        return best_state, best_reward


# from networkx documentation
def clique_number(g):
    """Returns the clique number of the graph.

    The clique number of a graph is the size of the largest clique in the graph."""
    return max(len(c) for c in nx.find_cliques(g))


# from grinpy source code
def _topological_index(G, func):
    """Return the topological index of ``G`` determined by ``func``"""

    return math.fsum(func(*edge) for edge in G.edges())


# from grinpy source code
def randic_index(G):
    r"""Returns the Randić Index of the graph ``G``.

    The *Randić index* of a graph *G* with edge set *E* is defined as the
    following sum:

    .. math::
        \sum_{vw \in E} \frac{1}{\sqrt{d_G(v) \times d_G(w)}}

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    float
        The Randić Index of a ``G``.

    References
    ----------

    Ivan Gutman, Degree-Based Topological Indices, Croat. Chem. Acta 86 (4)
    (2013) 351-361. http://dx.doi.org/10.5562/cca2294
    """
    _degree = functools.partial(nx.degree, G)
    return _topological_index(G, func=lambda x, y: 1 / math.sqrt(_degree(x) * _degree(y)))
