import argparse
import time
from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as npt
import tensorboardX
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import networkx as nx

import rewards
import utils

MIN_REWARD: int = -1000
"""Reward score for invalid graphs"""


def get_graph_word_size(nb_vertices: int) -> int:
    """
    Returns the number of edges in a graph with a given number of vertices.

    :param nb_vertices: number of vertices in the graph
    :return: number of edges in the graph
    """
    return nb_vertices * (nb_vertices - 1) // 2


def get_state_size(nb_vertices: int) -> int:
    """
    Returns the number of bits required to describe the state.

    The state is represented by n(n-1)/2 bits that describe the current
    edges of the graph, followed by n bits that describe the current
    edge being considered. Amongst these n bits exactly 2 are set to 1
    (the vertices of the edge being considered).

    :param nb_vertices: number of vertices in the graph
    :return: number of bits required to describe the state
    """
    return get_graph_word_size(nb_vertices) + nb_vertices


def make_model(
    nb_vertices: int,
    learning_rate: float,
    hidden_layer_neurons: List[int],
) -> keras.Model:
    """
    Creates a model to generate graphs.
    Input of the model is a state of the graph (current edges and
    description of edge being considered).
    Output is a probability of adding the edge being considered.

    :param nb_vertices: number of vertices in the graph
    :param learning_rate: learning rate of the model. Increase this to
    make convergence faster, decrease if the algorithm gets stuck in
    local optima too often.
    :param hidden_layer_neurons: number of neurons in the model hidden
    layers
    :return: the newly created model
    """
    state_size = get_state_size(nb_vertices)

    model: keras.Model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(shape=(state_size,)))
    for nb_neurons in hidden_layer_neurons:
        model.add(keras.layers.Dense(nb_neurons, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # Adam optimizer also works well, with lower learning rate
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
    )
    return model


def run_batch(
    nb_vertices: int,
    model: keras.Model,
    batch_size: int,
    reward_function: Callable[[npt.NDArray[np.bool_], int], float],
    action_randomness_epsilon: float = 0,
) -> Tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.float_],
]:
    """
    Generates a batch of graphs and evaluates the reward on the generated
    graphs.

    :param nb_vertices: number of vertices of each graph
    :param model: model used to generate the graphs
    :param batch_size: number of graphs to generate
    :param reward_function: reward function to evaluate the generated graphs
    :param action_randomness_epsilon: parameter to force some randomness.
    Whether an edge is added on the graph or not is based on the probability
    given by the model, but this probability is clipped to be at least
    (action_randomness_epsilon) and at most (1 - action_randomness_epsilon).
    :return: a tuple containing the generated graphs, states, actions and
    rewards
    """
    graph_word_size = get_graph_word_size(nb_vertices)
    state_size = get_state_size(nb_vertices)

    graphs: npt.NDArray[np.bool_] = np.zeros((batch_size, graph_word_size), dtype=bool)

    # states is an array of shape (batch_size, graph_word_size, state_size):
    # - coord 0: index in the batch
    # - coord 1: step of the generation process (one for each edge)
    # - value at (i, j): state of the i-th graph at step j (input of
    # the model represented as current edges followed by the description of
    # the active edge)
    states: npt.NDArray[np.bool_] = np.zeros((batch_size, graph_word_size, state_size), dtype=bool)

    # actions is an array of shape (batch_size, graph_word_size):
    # - coord 0: index in the batch
    # - coord 1: step of the generation process (one for each edge)
    # - value at (i, j): action taken  on the i-th graph at step j (1 if
    # an edge was added, 0 otherwise)
    actions: npt.NDArray[np.bool_] = np.zeros((batch_size, graph_word_size), dtype=bool)

    step_i = 0  # index of the first vertex of the edge being considered
    step_j = 1  # index of the second vertex of the edge being considered
    for step in range(graph_word_size):
        # for each possible edge of the graph, query the model to decide
        # whether to add it or not (on each of the batch_size graphs)
        states[:, step, :graph_word_size] = graphs
        states[:, step, graph_word_size + step_i] = 1
        states[:, step, graph_word_size + step_j] = 1

        prob = model(states[:, step, :]).numpy()  # model prediction
        prob = np.clip(prob, action_randomness_epsilon, 1 - action_randomness_epsilon)

        # choose action based on the model probability given by the model
        step_actions = (np.random.random(size=(batch_size,)) < prob[:, 0])
        actions[:, step] = step_actions
        # update graphs
        graphs[:, step] = step_actions

        step_j += 1
        if step_j == nb_vertices:
            step_i += 1
            step_j = step_i + 1

    # after generating all the graphs, calculate the reward for each of them
    rewards: npt.NDArray[np.float_] = np.apply_along_axis(reward_function, 1, graphs, nb_vertices)
    return graphs, states, actions, rewards


def run(
    nb_vertices: int,
    batch_size: int,
    reward_function: Callable[[npt.NDArray[np.bool_], int], float],
    elite_ratio: float,
    super_ratio: float,
    learning_rate: float,
    hidden_layer_neurons: List[int],
) -> None:
    """
    Executes the main loop of the learning process.

    :param nb_vertices: number of vertices in the graphs
    :param batch_size: number of new graphs generated per batch
    :param reward_function: reward function to evaluate the generated graphs
    :param elite_ratio: ratio of best instances to learn from
    :param super_ratio: ratio of best instances kept for the next iteration
    :param learning_rate: learning rate of the model. Increase this to make
    convergence faster, decrease if the algorithm gets stuck in local optima
    too often.
    :param hidden_layer_neurons: number of neurons in each hidden layer of
    the model
    """
    graph_word_size = get_graph_word_size(nb_vertices)
    state_size = get_state_size(nb_vertices)
    nb_elites = int(batch_size * elite_ratio)
    nb_supers = int(batch_size * super_ratio)
    model = make_model(nb_vertices, learning_rate, hidden_layer_neurons)

    best_reward: float = MIN_REWARD

    # initially there are no super instances
    super_states: npt.NDArray[np.bool_] = np.zeros(
        (0, graph_word_size, state_size), dtype=bool
    )
    super_actions: npt.NDArray[np.bool_] = np.zeros((0, graph_word_size), dtype=bool)
    super_rewards: npt.NDArray[np.float_] = np.zeros((0,), dtype=float)
    super_graphs: npt.NDArray[np.bool_] = np.zeros((0, graph_word_size), dtype=bool)

    iteration = 0
    steps_since_last_improvement = 0
    action_randomness_epsilon = 0.01

    writer = tensorboardX.SummaryWriter()
    while True:
        # generate a new batch of graphs
        graphs, states, actions, rewards = run_batch(
            nb_vertices,
            model,
            batch_size,
            reward_function,
            action_randomness_epsilon,
        )
        # append super graphs to the new batch
        states: npt.NDArray[np.bool_] = np.append(states, super_states, axis=0)
        actions: npt.NDArray[np.bool_] = np.append(actions, super_actions, axis=0)
        rewards: npt.NDArray[np.float_] = np.append(rewards, super_rewards)
        graphs: npt.NDArray[np.bool_] = np.append(graphs, super_graphs, axis=0)

        # get indexes of elites, supers and the best graph by reward
        indexes = np.argpartition(rewards, (-nb_elites, -nb_supers, -1))

        # Compare best reward of this batch to the overall best reward
        batch_best_reward = rewards[indexes[-1]]
        if batch_best_reward > best_reward:
            # log progress to tensorboard
            best_reward = batch_best_reward
            best_graph = graphs[indexes[-1]]
            plt.figure(num=1, figsize=(4, 4), dpi=300)
            nx.draw_kamada_kawai(
                utils.make_graph(best_graph, nb_vertices), node_size=80
            )
            writer.add_figure("best graph", plt.figure(num=1), iteration)
            writer.add_text(
                "best matrix", repr(best_graph[:graph_word_size]), iteration
            )
            plt.close()
            writer.flush()

            steps_since_last_improvement = 0
            action_randomness_epsilon = 0.01
        else:
            steps_since_last_improvement += 1
            if steps_since_last_improvement % 10 == 0:
                action_randomness_epsilon *= 1.1
                action_randomness_epsilon = min(action_randomness_epsilon, 0.1)

        # evaluate mean reward and log to tensorboard
        iteration += 1
        writer.add_scalar("reward/best", best_reward, iteration)
        writer.add_scalar("reward/mean", np.mean(rewards), iteration)
        writer.add_scalar("action randomness", action_randomness_epsilon, iteration)
        writer.flush()

        # select elites (sessions to learn from)
        elite_indexes = indexes[-nb_elites:]
        elite_states = states[elite_indexes]
        elite_actions = actions[elite_indexes]

        model.fit(elite_states.reshape(-1, state_size), elite_actions.reshape(-1))
        keras.backend.clear_session()  # to mitigate memory leak

        # select super sessions (sessions that will be kept for the next generation)
        super_indexes = indexes[-nb_supers:]
        super_states = states[super_indexes]
        super_actions = actions[super_indexes]
        super_rewards = rewards[super_indexes]
        super_graphs = graphs[super_indexes]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "reward_function",
        type=str,
        help=f"reward function to use. Possible values: {', '.join(rewards.mapping.keys())}",
    )
    parser.add_argument(
        "nb_vertices",
        type=int,
        help="number of vertices in the graph",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        type=float,
        default=0.0005,
        help="learning rate of the model. Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1000,
        help="number of new sessions per iteration",
    )
    parser.add_argument(
        "--elite-ratio",
        type=float,
        default=0.1,
        help="ratio of best instances we are learning from",
    )
    parser.add_argument(
        "--super-ratio",
        type=float,
        default=0.05,
        help="ratio of best instances that survive to the next iteration",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[64, 32],
        help="number of neurons in the model hidden layers",
    )

    args = parser.parse_args()

    get_reward = rewards.mapping[args.reward_function]
    run(
        nb_vertices=args.nb_vertices,
        batch_size=args.batch_size,
        elite_ratio=args.elite_ratio,
        super_ratio=args.super_ratio,
        learning_rate=args.learning_rate,
        hidden_layer_neurons=args.hidden_layers,
        reward_function=get_reward,
    )
