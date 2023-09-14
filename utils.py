import numpy as np
import networkx as nx


EPSILON = 1e-10
INF = float("inf")


def get_nb_vertices(nb_edges: int) -> int:
    return int((1 + np.sqrt(1 + 8 * nb_edges)) / 2)


def laplacian(m: np.ndarray[int]) -> np.ndarray[int]:
    """Returns the laplacian of a graph given by its adjacency matrix.

    The laplacian of a graph is defined as the difference between the degree matrix and the adjacency matrix.
    """

    l = -m
    for i in range(len(m)):
        l[i, i] = m[i, :].sum()
    return l


def signless_laplacian(m: np.ndarray[int]) -> np.ndarray[int]:
    """
    Returns the signless laplacian of a graph given by its adjacency matrix.

    The signless laplacian of a graph is defined as the sum of the degree matrix and the adjacency matrix.
    """
    l = m[:, :]
    for i in range(len(l)):
        l[i, i] = m[i, :].sum()
    return l


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
