import argparse
import time
from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as npt
import tensorboardX
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx

import rewards
import utils

INF = 1000


def get_graph_word_size(nb_vertices: int) -> int:
    return nb_vertices * (nb_vertices - 1) // 2


def get_state_size(nb_vertices: int) -> int:
    return get_graph_word_size(nb_vertices) + nb_vertices


def make_model(
    nb_vertices: int,
    learning_rate: float,
    hidden_layer_neurons: List[int],
) -> tf.keras.Model:
    state_size = get_state_size(nb_vertices)

    model: tf.keras.Model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(state_size,)))
    for nb_neurons in hidden_layer_neurons:
        model.add(tf.keras.layers.Dense(nb_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # Adam optimizer also works well, with lower learning rate
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
    )
    return model


def run_batch(
    nb_vertices: int,
    model: tf.keras.Model,
    batch_size: int,
    best_graphs: npt.NDArray,
    get_reward: Callable[[npt.NDArray, int], float],
    action_randomness_epsilon: float = 0,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    graph_word_size = get_graph_word_size(nb_vertices)
    state_size = get_state_size(nb_vertices)

    graphs = np.repeat(best_graphs, batch_size // len(best_graphs), axis=0)
    graphs = np.append(graphs, best_graphs[: batch_size % len(best_graphs)], axis=0)

    states = np.zeros((batch_size, graph_word_size, state_size), dtype=int)
    actions = np.zeros((batch_size, graph_word_size), dtype=int)

    step_i = 0
    step_j = 1
    step_k = 0
    for step in range(graph_word_size):
        states[:, step, :graph_word_size] = graphs
        states[:, step, graph_word_size + step_i] = 1
        states[:, step, graph_word_size + step_j] = 1

        prob = model(states[:, step, :]).numpy()
        prob = np.clip(prob, action_randomness_epsilon, 1 - action_randomness_epsilon)

        step_actions = (np.random.random(size=(batch_size,)) < prob[:, 0]).astype(int)
        actions[:, step] = step_actions
        # update graphs
        graphs[:, step_k] = step_actions

        step_j += 1
        if step_j == nb_vertices:
            step_i += 1
            step_j = step_i + 1
        step_k += 1

    rewards = np.apply_along_axis(get_reward, 1, graphs, nb_vertices)

    return graphs, states, actions, rewards


def run(
    nb_vertices: int,
    batch_size: int,
    get_reward: Callable[[npt.NDArray, int], float],
    elite_ratio: float,
    super_ratio: float,
    learning_rate: float,
    hidden_layer_neurons: List[int],
) -> None:
    """
    Runs the learning process

    :param nb_vertices: number of vertices in the graph
    :param batch_size: number of new sessions per iteration
    :param elite_ratio: ratio of best instances we are learning from
    :param super_ratio: ratio of best instances that survive to the next iteration
    :param learning_rate: learning rate of the model. Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
    :param hidden_layer_neurons: number of neurons in the model hidden layers
    """
    graph_word_size = get_graph_word_size(nb_vertices)
    state_size = get_state_size(nb_vertices)
    nb_elites = int(batch_size * elite_ratio)
    nb_supers = int(batch_size * super_ratio)

    best_reward: float = -INF
    start_time = time.time()

    model = make_model(nb_vertices, learning_rate, hidden_layer_neurons)

    super_states = np.zeros((0, graph_word_size, state_size), dtype=int)
    super_actions = np.zeros((0, graph_word_size), dtype=int)
    super_rewards = np.zeros((0,), dtype=float)
    super_graphs = np.zeros((0, graph_word_size), dtype=int)

    best_graphs = np.random.randint(2, size=(batch_size, graph_word_size))
    iteration = 0

    steps_since_last_improvement = 0
    action_randomness_epsilon = .01

    writer = tensorboardX.SummaryWriter()
    while True:
        # generate new sessions
        tic = time.time()
        graphs, states, actions, rewards = run_batch(
            nb_vertices,
            model,
            batch_size,
            best_graphs,
            get_reward,
            action_randomness_epsilon,
        )
        states = np.append(states, super_states, axis=0)
        actions = np.append(actions, super_actions, axis=0)
        rewards = np.append(rewards, super_rewards)
        graphs = np.append(graphs, super_graphs, axis=0)

        # Compare best reward of this batch to the overall best reward
        batch_best_reward_index = np.argmax(rewards)
        batch_best_reward = rewards[batch_best_reward_index]
        if batch_best_reward > best_reward:
            # log progress to tensorboard
            best_reward = batch_best_reward
            best_graph = graphs[batch_best_reward_index]
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
            action_randomness_epsilon = .01
        else:
            steps_since_last_improvement += 1
            if steps_since_last_improvement > 10:
                action_randomness_epsilon *= 1.2
                steps_since_last_improvement = 0
                action_randomness_epsilon = min(action_randomness_epsilon, .1)

        # select elites (sessions to learn from)
        indexes = np.argpartition(rewards, (-nb_elites, -nb_supers))
        elite_indexes = indexes[-nb_elites:]
        elite_states = states[elite_indexes].reshape(-1, state_size)
        elite_actions = actions[elite_indexes].reshape(-1)

        best_graphs = graphs[elite_indexes]

        # select super sessions (sessions that will be kept for the next generation)
        super_indexes = indexes[-nb_supers:]
        super_states = states[super_indexes]
        super_actions = actions[super_indexes]
        super_rewards = rewards[super_indexes]
        super_graphs = graphs[super_indexes]

        model.fit(elite_states, elite_actions)
        tf.keras.backend.clear_session()  # to mitigate memory leak

        # evaluate mean reward
        mean_all_reward = np.mean(rewards)

        # print(f"{iteration}. Best individuals:")
        # print(np.sort(super_rewards)[::-1])
        # print(f"Mean reward: {mean_all_reward}, time: {time.time() - tic}")
        # print()
        iteration += 1
        writer.add_scalar("reward/best", best_reward, iteration)
        writer.add_scalar("reward/mean", mean_all_reward, iteration)
        writer.add_scalar("action randomness", action_randomness_epsilon, iteration)
        writer.flush()

        # if iteration % 50 == 0:
        #     # save model every 50 iterations
        #     model.save("./model.keras")


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
        default=2000,
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
        default=[128, 32],
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
        get_reward=get_reward,
    )
