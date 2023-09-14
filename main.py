import argparse
import random
import time
from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf

import rewards

INF = float("inf")


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
    game_length: int,
    model: tf.keras.Model,
    batch_size: int,
    best_graphs: npt.NDArray,
    get_reward: Callable[[npt.NDArray, int], float],
    random_order: bool,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    graph_word_size = get_graph_word_size(nb_vertices)
    state_size = get_state_size(nb_vertices)

    graphs = np.repeat(best_graphs, batch_size // len(best_graphs), axis=0)
    graphs = np.append(graphs, best_graphs[: batch_size % len(best_graphs)], axis=0)

    states = np.zeros((batch_size, game_length, state_size), dtype=int)
    actions = np.zeros((batch_size, game_length), dtype=int)

    if not random_order:
        step_i = 0
        step_j = 1
        step_k = 0
    for step in range(game_length):
        states[:, step, :graph_word_size] = graphs
        if random_order:
            # pick a random edge
            step_i = random.randrange(nb_vertices)
            step_j = (step_i + random.randrange(nb_vertices - 1)) % nb_vertices
            step_i, step_j = min(step_i, step_j), max(step_i, step_j)
            step_k = (
                nb_vertices * step_i - (step_i * (step_i + 1)) // 2 + step_j - step_i - 1
            )

        states[:, step, graph_word_size + step_i] = 1
        states[:, step, graph_word_size + step_j] = 1

        prob = model(states[:, step, :]).numpy()
        step_actions = (np.random.random(size=(batch_size,)) < prob[:, 0]).astype(int)
        actions[:, step] = step_actions
        # update graphs
        graphs[:, step_k] = step_actions

        if not random_order:
            step_j += 1
            if step_j == nb_vertices:
                step_i += 1
                step_j = step_i + 1
            step_k += 1

    rewards = np.apply_along_axis(get_reward, 1, graphs, nb_vertices)

    return graphs, states, actions, rewards


def run(
    nb_vertices: int,
    game_length: int,
    batch_size: int,
    get_reward: Callable[[npt.NDArray, int], float],
    elite_ratio: float,
    super_ratio: float,
    learning_rate: float,
    hidden_layer_neurons: List[int],
    output_file: str,
    random_order: bool,
) -> None:
    """
    Runs the learning process

    :param nb_vertices: number of vertices in the graph
    :param game_length: length of the game
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
    if game_length is None:
        game_length = nb_vertices * (nb_vertices - 1) // 2

    best_reward: float = -INF
    start_time = time.time()

    model = make_model(nb_vertices, learning_rate, hidden_layer_neurons)

    super_states = np.zeros((0, game_length, state_size), dtype=int)
    super_actions = np.zeros((0, game_length), dtype=int)
    super_rewards = np.zeros((0,), dtype=float)
    super_graphs = np.zeros((0, graph_word_size), dtype=int)

    best_graphs = np.random.randint(2, size=(batch_size, graph_word_size))
    iteration = 0

    while True:
        # generate new sessions
        tic = time.time()
        graphs, states, actions, rewards = run_batch(
            nb_vertices,
            game_length,
            model,
            batch_size,
            best_graphs,
            get_reward,
            random_order,
        )
        states = np.append(states, super_states, axis=0)
        actions = np.append(actions, super_actions, axis=0)
        rewards = np.append(rewards, super_rewards)
        graphs = np.append(graphs, super_graphs, axis=0)

        # select elites (sessions to learn from)
        indexes = np.argpartition(rewards, (-nb_elites, -nb_supers))
        elite_indexes = indexes[-nb_elites:]
        elite_states = states[elite_indexes].reshape(-1, state_size)
        elite_actions = actions[elite_indexes].reshape(-1)

        # Compare best reward of this batch to the overall best reward
        batch_best_reward_index = np.argmax(rewards)
        batch_best_reward = rewards[batch_best_reward_index]
        if batch_best_reward > best_reward:
            best_reward = batch_best_reward
            best_graph = graphs[batch_best_reward_index]
            with open(output_file, "a") as f:
                f.write("".join(str(x) for x in best_graph) + "\n")
                f.write(f"{best_reward} ({time.time() - start_time})\n\n")

        model.fit(elite_states, elite_actions)
        tf.keras.backend.clear_session()  # to mitigate memory leak

        best_graphs = graphs[elite_indexes]

        # select super sessions (sessions that will be kept for the next generation)
        super_indexes = indexes[-nb_supers:]
        super_states = states[super_indexes]
        super_actions = actions[super_indexes]
        super_rewards = rewards[super_indexes]
        super_graphs = graphs[super_indexes]

        # evaluate mean reward
        mean_all_reward = np.mean(rewards)

        print(f"{iteration}. Best individuals:")
        print(np.sort(super_rewards)[::-1])
        print(f"Mean reward: {mean_all_reward}, time: {time.time() - tic}")
        print()
        iteration += 1

        if iteration % 50 == 0:
            # save model every 50 iterations
            model.save("./model.keras")


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
        "--game-length",
        "-g",
        type=int,
        default=None,
        help="length of a game (number of times a random edge can be changed)",
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
        default=[128, 64, 4],
        help="number of neurons in the model hidden layers",
    )
    parser.add_argument(
        "--random-order",
        action="store_true",
        help="randomize the order of edges in each game",
        default=False,
    )

    args = parser.parse_args()

    get_reward = rewards.mapping[args.reward_function]
    run(
        nb_vertices=args.nb_vertices,
        game_length=args.game_length,
        batch_size=args.batch_size,
        elite_ratio=args.elite_ratio,
        super_ratio=args.super_ratio,
        learning_rate=args.learning_rate,
        hidden_layer_neurons=args.hidden_layers,
        output_file=args.output_file,
        get_reward=get_reward,
        random_order=args.random_order,
    )
