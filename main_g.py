import argparse
import random
import time
from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf

import rewards
import utils

INF = float("inf")


def get_graph_word_size(nb_vertices: int) -> int:
    return nb_vertices * (nb_vertices - 1) // 2


def get_state_size(nb_vertices: int) -> int:
    return get_graph_word_size(nb_vertices) + nb_vertices


def make_models(
    nb_vertices: int,
    learning_rate: float,
    hidden_layer_neurons: List[int],
) -> tf.keras.Model:
    nb_edges = get_graph_word_size(nb_vertices)
   
    models = []
    for i in range(nb_edges):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(i,)))
        for nb_neurons in hidden_layer_neurons:
            model.add(tf.keras.layers.Dense(nb_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        # Adam optimizer also works well, with lower learning rate
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate),
        )
        models.append(model)
    return models


def run_batch(
    nb_vertices: int,
    models: [tf.keras.Model],
    batch_size: int,
    get_reward: Callable[[npt.NDArray, int], float],
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    nb_edges = get_graph_word_size(nb_vertices)
    graphs = np.zeros((batch_size, nb_edges), dtype=int)

    for step in range(nb_edges):
        prob = models[step](graphs[:, :step]).numpy()
        step_actions = (np.random.random(size=(batch_size,)) < prob[:, 0]).astype(int)
        graphs[:, step] = step_actions

    rewards = np.apply_along_axis(get_reward, 1, graphs, nb_vertices)
    return graphs, rewards


def run(
    nb_vertices: int,
    batch_size: int,
    get_reward: Callable[[npt.NDArray, int], float],
    elite_ratio: float,
    super_ratio: float,
    learning_rate: float,
    hidden_layer_neurons: List[int],
    output_file: str,
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
    nb_edges = get_graph_word_size(nb_vertices)
    nb_elites = int(batch_size * elite_ratio)
    nb_supers = int(batch_size * super_ratio)

    best_reward: float = -INF
    start_time = time.time()

    models = make_models(nb_vertices, learning_rate, hidden_layer_neurons)

    super_rewards = np.full((0,), -INF, dtype=float)
    super_graphs = np.zeros((0, nb_edges), dtype=int)

    best_graphs = np.random.randint(2, size=(batch_size, nb_edges))
    iteration = 0

    while True:
        # generate new sessions
        tic = time.time()
        graphs, rewards = run_batch(
            nb_vertices,
            models,
            batch_size,
            get_reward,
        )
        rewards = np.append(rewards, super_rewards)
        graphs = np.append(graphs, super_graphs, axis=0)

        # Compare best reward of this batch to the overall best reward
        batch_best_reward_index = np.argmax(rewards)
        batch_best_reward = rewards[batch_best_reward_index]
        if batch_best_reward > best_reward:
            best_reward = batch_best_reward
            best_graph = graphs[batch_best_reward_index]
            with open(output_file, "a") as f:
                f.write("".join(str(x) for x in best_graph) + "\n")
                f.write(f"{best_reward} ({time.time() - start_time})\n\n")

        # select elites (sessions to learn from)
        indexes = np.argpartition(rewards, (-nb_elites, -nb_supers))
        elite_indexes = indexes[-nb_elites:]
        elite_graphs = graphs[elite_indexes]
        # elite_states = states[elite_indexes].reshape(-1, nb_edges)
        # elite_actions = actions[elite_indexes].reshape(-1)


        # select super sessions (sessions that will be kept for the next generation)
        super_indexes = indexes[-nb_supers:]
        super_rewards = rewards[super_indexes]
        super_graphs = graphs[super_indexes]

        for i in range(nb_edges):
            models[i].fit(elite_graphs[:,:i], elite_graphs[:,i])
        tf.keras.backend.clear_session()  # to mitigate memory leak

        # evaluate mean reward
        mean_all_reward = np.mean(rewards)

        print(f"{iteration}. Best individuals:")
        print(np.sort(super_rewards)[::-1])
        print(f"Mean reward: {mean_all_reward}, time: {time.time() - tic}")
        print()
        iteration += 1

        # if iteration % 50 == 0:
        #     # save model every 50 iterations
        #     for i, model in enumerate(models):
        #         model[i].save("./model.keras")


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
        "--output-file",
        "-o",
        type=str,
        default="results.txt",
        help="file to write results to",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        type=float,
        default=0.0001,
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
        default=0.07,
        help="ratio of best instances we are learning from",
    )
    parser.add_argument(
        "--super-ratio",
        type=float,
        default=0.06,
        help="ratio of best instances that survive to the next iteration",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[8, 8, 4],
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
        output_file=args.output_file,
        get_reward=get_reward,
    )
