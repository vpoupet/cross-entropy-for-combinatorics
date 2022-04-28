import random
import time

import numpy as np
import tensorflow

N = 19
MYN = int(N * (N - 1) / 2)

# Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
LEARNING_RATE = 0.0001
BATCH_SIZE = 1000  # number of new sessions per iteration
NB_ELITE = 100

FIRST_LAYER_NEURONS = 128  # Number of neurons in the hidden layers.
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

NB_ACTIONS = 2
STATE_SIZE = MYN + N
GAME_LENGTH = MYN

# Construction of the prediction model
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(FIRST_LAYER_NEURONS, activation="relu"))
model.add(tensorflow.keras.layers.Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(tensorflow.keras.layers.Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(tensorflow.keras.layers.Dense(1, activation="sigmoid"))
model.build((None, STATE_SIZE))
model.compile(
    loss="binary_crossentropy",
    optimizer=tensorflow.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
)

print(model.summary())


def get_reward(state):
    g = np.zeros((N, N), dtype=int)
    k = 0
    for i in range(N - 1):
        g[i, i + 1 :] = state[k : k + N - i - 1]
        k += N - i - 1
    return -abs(g.sum(axis=-1) - 3).sum()


def run_batch(model, batch_size, graphs=None):
    states = np.zeros((batch_size, GAME_LENGTH, STATE_SIZE), dtype=int)
    actions = np.zeros([batch_size, GAME_LENGTH], dtype=int)
    if graphs is not None:
        states[:, 0, :MYN] = np.repeat(graphs, batch_size // graphs.shape[0], axis=0)
    prob = np.zeros(batch_size)
    step_i = 0
    step_j = 1
    rewards = np.zeros((batch_size,))

    for step in range(GAME_LENGTH):
        for i in range(batch_size):
            step_i = random.randrange(N)
            step_j = (step_i + random.randrange(N - 1)) % N
            step_k = (step_i * (step_i + 1)) // 2 + step_j
        
            states[i, step, MYN + step_i] = 1
            states[i, step, MYN + step_j] = 1
        prob = model.predict(states[:, step, :], batch_size=batch_size)

        if step > 0:
            states[:, step, :] = states[:, step - 1, :]
        actions[:, step] = np.random.random(size=(batch_size,)) < prob[:, 0]
        states[:, step, step_k] = actions[:, step]

    for i in range(batch_size):
        rewards[i] = get_reward(states[i, GAME_LENGTH - 1, :])

    return states, actions, rewards


if __name__ == "__main__":
    best_graphs = np.random.randint(2, size=(BATCH_SIZE, MYN))
    for i in range(1000000):
        tic = time.time()
        states, actions, rewards = run_batch(model, BATCH_SIZE, best_graphs)

        # select elites (sessions to learn from)
        elite_indexes = np.argpartition(rewards, -NB_ELITE)[-NB_ELITE:]
        elite_states = np.concatenate(states[elite_indexes])
        elite_actions = np.concatenate(actions[elite_indexes])
        model.fit(np.concatenate(states), np.concatenate(actions))

        best_graphs = states[elite_indexes, GAME_LENGTH - 1, :MYN]

        mean_all_reward = np.mean(rewards)
        nb_best = 10
        best_indexes = np.argpartition(rewards, -nb_best)[-nb_best:]
        best_states = np.concatenate(states[best_indexes])
        best_actions = np.concatenate(actions[best_indexes])

        print(f"{i}. Best individuals:")
        print(np.sort(rewards)[: -nb_best - 1 : -1])
        print(f"Mean reward: {mean_all_reward}, time: {time.time() - tic}")
        print()
