import random
import time
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow

from utils import make_graph, EPSILON
from rewards import get_reward_third_eigenvalue as get_reward


def get_graph_word_size(nb_vertices: int) -> int:
    return nb_vertices * (nb_vertices - 1) // 2


def get_state_size(nb_vertices: int) -> int:
    return get_graph_word_size(nb_vertices) + nb_vertices


def make_model(
    nb_vertices: int,
    learning_rate: float,
    hidden_layer_neurons: List[int],
) -> tensorflow.keras.Model:
    state_size = get_state_size(nb_vertices)

    model: tensorflow.keras.Model = tensorflow.keras.models.Sequential()
    for nb_neurons in hidden_layer_neurons:
        model.add(tensorflow.keras.layers.Dense(nb_neurons, activation="relu"))
    model.add(tensorflow.keras.layers.Dense(1, activation="sigmoid"))
    model.build((None, state_size))
    # Adam optimizer also works well, with lower learning rate
    model.compile(
        loss="binary_crossentropy",
        optimizer=tensorflow.keras.optimizers.SGD(learning_rate=learning_rate),
    )
    print(model.summary())
    return model


# No need to change anything below here.


def run_batch(
    nb_vertices: int,
    model: tensorflow.keras.Model,
    batch_size: int,
    game_length: int,
    graphs: np.ndarray = None,
) -> None:
    graph_word_size = get_graph_word_size(nb_vertices)
    state_size = get_state_size(nb_vertices)

    states: np.ndarray[int] = np.zeros((batch_size, game_length, state_size), dtype=int)
    actions: np.ndarray[int] = np.zeros([batch_size, game_length], dtype=int)
    if graphs is not None:
        states[:, 0, :graph_word_size] = np.repeat(
            graphs, batch_size // graphs.shape[0], axis=0
        )

    prob = np.zeros(batch_size)
    step_i: int = 0
    step_j: int = 1
    rewards = np.zeros((batch_size,))

    for step in range(game_length):
        step_i = random.randrange(nb_vertices)
        step_j = (step_i + random.randrange(nb_vertices - 1)) % nb_vertices
        states[:, step, graph_word_size + step_i] = 1
        states[:, step, graph_word_size + step_j] = 1
        step_k = (step_i * (step_i + 1)) // 2 + step_j
        prob = model.predict(states[:, step, :], batch_size=batch_size)

        if step > 0:
            states[:, step, :graph_word_size] = states[:, step - 1, :graph_word_size]
        actions[:, step] = np.random.random(size=(batch_size,)) < prob[:, 0]
        states[:, step, step_k] = actions[:, step]

    for i in range(batch_size):
        rewards[i] = get_reward(states[i, game_length - 1, :graph_word_size], nb_vertices)

    return states, actions, rewards


def run(
    nb_vertices: int,
    game_length: int,
    batch_size: int,
    elite_ratio: float,
    super_ratio: float,
    learning_rate: float,
    hidden_layer_neurons: List[int],
    output_file: str,
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

    best_reward: float = -10
    start_time = time.time()

    model = make_model(nb_vertices, learning_rate, hidden_layer_neurons)

    super_states = np.zeros((0, game_length, state_size), dtype=int)
    super_actions = np.zeros((0, game_length), dtype=int)
    super_rewards = np.zeros((0,))

    best_graphs = np.random.randint(2, size=(batch_size, graph_word_size))
    iteration = 0

    while True:
        # generate new sessions
        tic = time.time()
        states, actions, rewards = run_batch(
            nb_vertices, model, batch_size, game_length, graphs=best_graphs
        )
        states = np.append(states, super_states, axis=0)
        actions = np.append(actions, super_actions, axis=0)

        states = np.append(states, super_states, axis=0)
        actions = np.append(actions, np.array(super_actions), axis=0)
        rewards = np.append(rewards, super_rewards)

        # select elites (sessions to learn from)
        elite_indexes = np.argpartition(rewards, -nb_elites)[-nb_elites:]
        elite_states = np.concatenate(states[elite_indexes])
        elite_actions = np.concatenate(actions[elite_indexes])

        # Compare best reward of this batch to the overall best reward
        batch_best_reward_index = np.argmax(rewards)
        batch_best_reward = rewards[batch_best_reward_index]
        if batch_best_reward > best_reward:
            best_reward = batch_best_reward
            best_graph = states[
                batch_best_reward_index, game_length - 1, :graph_word_size
            ]
            with open(output_file, "a") as f:
                f.write("".join(str(x) for x in best_graph) + "\n")
                f.write(f"{best_reward} ({time.time() - start_time})\n\n")
            if best_reward > EPSILON:
                print(f"Best reward: {best_reward}")
                # print best instance found
                print(f"[{', '.join(str(x) for x in best_graph)}]")
                nx.draw_kamada_kawai(make_graph(best_graph, nb_vertices))
                plt.show()
                exit()

        model.fit(elite_states, elite_actions)
        best_graphs = states[elite_indexes, game_length - 1, :graph_word_size]

        # select super sessions (sessions that will be kept for the next generation)
        super_indexes = np.argpartition(rewards, -nb_supers)[-nb_supers:]
        super_states = states[super_indexes]
        super_actions = actions[super_indexes]
        super_rewards = rewards[super_indexes]

        # evaluate mean reward
        mean_all_reward = np.mean(rewards)

        print(f"{iteration}. Best individuals:")
        print(np.sort(super_rewards)[::-1])
        print(f"Mean reward: {mean_all_reward}, time: {time.time() - tic}")
        print()
        iteration += 1


if __name__ == "__main__":
    run(
        nb_vertices=20,
        game_length=20,
        batch_size=2000,
        elite_ratio=0.1,
        super_ratio=0.05,
        learning_rate=0.0005,
        hidden_layer_neurons=[128, 64, 4],
        output_file="results.txt",
    )