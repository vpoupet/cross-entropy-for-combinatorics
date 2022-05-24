import math
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow

N = 25  # number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm
# The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)
MYN = N * (N - 1) // 2

# Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
LEARNING_RATE = 0.001
BATCH_SIZE = 1000  # number of new sessions per iteration
ELITE_PERCENTILE = 90  # top 100-X percentile we are learning from
SUPER_PERCENTILE = 95  # top 100-X percentile that survives to next iteration
NB_ELITE = int(BATCH_SIZE * (100 - ELITE_PERCENTILE) / 100)
NB_SUPER = int(BATCH_SIZE * (100 - SUPER_PERCENTILE) / 100)

FIRST_LAYER_NEURONS = 128  # Number of neurons in the hidden layers.
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4

NB_ACTIONS = 2  # The size of the alphabet. In this file we will assume this is 2. There are a few things we need to change when the alphabet size is larger,
# such as one-hot encoding the input, and using categorical_crossentropy as a loss function.

# Leave this at 2*MYN. The input vector will have size 2*MYN, where the first MYN letters encode our partial word (with zeros on
STATE_SIZE = MYN + N
# the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
# So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
# Is there a better way to format the input to make it easier for the neural network to understand things?
GAME_LENGTH = 10

EPS = 1e-8
INF = 10000

# Model structure: a sequential network with three hidden layers, sigmoid activation in the output.
# I usually used relu activation in the hidden layers but play around to see what activation function and what optimizer works best.
# It is important that the loss is binary cross-entropy if alphabet size is 2.

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(FIRST_LAYER_NEURONS, activation="relu"))
model.add(tensorflow.keras.layers.Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(tensorflow.keras.layers.Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(tensorflow.keras.layers.Dense(1, activation="sigmoid"))
model.build((None, STATE_SIZE))
# Adam optimizer also works well, with lower learning rate
model.compile(
    loss="binary_crossentropy",
    optimizer=tensorflow.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
)

print(model.summary())


def laplacian(g):
    l = -g
    for i in range(len(g)):
        l[i, i] = g[i, :].sum()
    return l


def signless_laplacian(g):
    l = g[:, :]
    for i in range(len(l)):
        l[i, i] = g[i, :].sum()
    return l


def nb_components(g):
    l = laplacian(g)
    eigen_values = np.linalg.eigvals(l)
    return len(eigen_values) - np.count_nonzero(eigen_values)


def make_graph(state):
    g = nx.Graph()
    g.add_nodes_from(list(range(N)))
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if state[count] == 1:
                g.add_edge(i, j)
            count += 1
    return g


def make_matrix(state):
    g_array = np.zeros((N, N), dtype=int)
    k = 0
    for i in range(N - 1):
        g_array[i, i + 1 :] = state[k : k + N - i - 1]
        g_array[i + 1 :, i] = state[k : k + N - i - 1]
        k += N - i - 1
    return g_array


def get_reward_avg_deg(state):
    return -abs(3 - np.sum(state[:MYN]) * 2 / N)


def get_reward_tree(state):
    g = make_matrix(state)
    if nb_components(g) > 1:
        return -100
    return -sum(state[:MYN])


def get_reward_deg(state):
    g = make_matrix(state)
    return -abs(g.sum(axis=-1) - 3).sum()


def get_reward_brouwer(state):
    global best_reward, best_state

    t = 10
    g = make_matrix(state)
    l = laplacian(g)
    eigen_values = np.sort(np.linalg.eigvals(l))
    return sum(eigen_values[-t:]) - np.count_nonzero(state[:MYN]) - t * (t + 1) / 2


def get_reward_ashraf(state):
    global best_reward, best_state

    t = 10
    g = make_matrix(state)
    l = signless_laplacian(g)
    eigen_values = np.sort(np.linalg.eigvals(l))
    return sum(eigen_values[-t:]) - np.count_nonzero(state[:MYN]) - t * (t + 1) / 2


def get_reward_conj21(state):
    """
    Calculates the reward for a given word.
    This function is very slow, it can be massively sped up with numba -- but numba doesn't support networkx yet, which is very convenient to use here

    :param state: the first MYN letters of this param are the word that the neural network has constructed.


    :returns: the reward (a real number). Higher is better, the network will try to maximize this.
    """

    # Example reward function, for Conjecture 2.1
    # Given a graph, it minimizes lambda_1 + mu.
    # Takes a few hours  (between 300 and 10000 iterations) to converge (loss < 0.01) on my computer with these parameters if not using parallelization.
    # There is a lot of run-to-run variance.
    # Finds the counterexample some 30% (?) of the time with these parameters, but you can run several instances in parallel.

    # Construct the graph
    g_array = np.zeros((N, N), dtype=int)
    k = 0
    for i in range(N - 1):
        g_array[i, i + 1 :] = state[k : k + N - i - 1]
        k += N - i - 1

    g = nx.from_numpy_matrix(g_array)

    # g is assumed to be connected in the conjecture. If it isn't, return a very negative reward.
    if not (nx.is_connected(g)):
        return -INF

    # Calculate the eigenvalues of g
    evals = np.linalg.eigvalsh(nx.adjacency_matrix(g).todense())
    evalsRealAbs = np.zeros_like(evals)
    for i in range(len(evals)):
        evalsRealAbs[i] = abs(evals[i])
    lambda1 = max(evalsRealAbs)

    # Calculate the matching number of g
    maxMatch = nx.max_weight_matching(g)
    mu = len(maxMatch)

    # Calculate the reward. Since we want to minimize lambda_1 + mu, we return the negative of this.
    # We add to this the conjectured best value. This way if the reward is positive we know we have a counterexample.
    reward = math.sqrt(N - 1) + 1 - lambda1 - mu
    if reward > 0:
        # You have found a counterexample. Do something with it.
        print(state)
        nx.draw_kamada_kawai(g)
        plt.show()
        exit()

    return reward


get_reward = get_reward_ashraf

# No need to change anything below here.


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
        step_i = random.randrange(N)
        step_j = (step_i + random.randrange(N - 1)) % N
        states[:, step, MYN + step_i] = 1
        states[:, step, MYN + step_j] = 1
        step_k = (step_i * (step_i + 1)) // 2 + step_j
        prob = model.predict(states[:, step, :], batch_size=batch_size)

        if step > 0:
            states[:, step, :MYN] = states[:, step - 1, :MYN]
        actions[:, step] = np.random.random(size=(batch_size,)) < prob[:, 0]
        states[:, step, step_k] = actions[:, step]

    for i in range(batch_size):
        rewards[i] = get_reward(states[i, GAME_LENGTH - 1, :])

    return states, actions, rewards


if __name__ == "__main__":
    best_reward = -1
    best_graph = None
    start_time = time.time()

    super_states = np.zeros((0, GAME_LENGTH, STATE_SIZE), dtype=int)
    super_actions = np.zeros((0, GAME_LENGTH), dtype=int)
    super_rewards = np.zeros((0,))

    myRand = random.randint(0, 1000)  # used in the filename

    best_graphs = np.random.randint(2, size=(BATCH_SIZE, MYN))
    iteration = 0

    while True:
        # generate new sessions
        tic = time.time()
        states, actions, rewards = run_batch(model, BATCH_SIZE, graphs=best_graphs)
        states = np.append(states, super_states, axis=0)
        actions = np.append(actions, super_actions, axis=0)

        states = np.append(states, super_states, axis=0)
        actions = np.append(actions, np.array(super_actions), axis=0)
        rewards = np.append(rewards, super_rewards)

        # select elites (sessions to learn from)
        elite_indexes = np.argpartition(rewards, -NB_ELITE)[-NB_ELITE:]
        elite_states = np.concatenate(states[elite_indexes])
        elite_actions = np.concatenate(actions[elite_indexes])

        # Compare best reward of this batch to the overall best reward
        batch_best_reward_index = np.argmax(rewards)
        batch_best_reward = rewards[batch_best_reward_index]
        if batch_best_reward > best_reward:
            best_reward = batch_best_reward
            best_graph = states[batch_best_reward_index, GAME_LENGTH - 1, :MYN]
            with open("results.txt", "a") as f:
                f.write("".join(str(x) for x in best_graph) + "\n")
                f.write(f"{best_reward} ({time.time() - start_time})\n\n")
            if best_reward > EPS:
                print(best_graph)
                nx.draw_kamada_kawai(make_graph(best_graph))
                plt.show()
                exit()

        model.fit(elite_states, elite_actions)
        best_graphs = states[elite_indexes, GAME_LENGTH - 1, :MYN]

        # select super sessions (sessions that will be kept for the next generation)
        super_indexes = np.argpartition(rewards, -NB_SUPER)[-NB_SUPER:]
        super_states = states[super_indexes]
        super_actions = actions[super_indexes]
        super_rewards = rewards[super_indexes]

        # evaluate mean reward
        mean_all_reward = np.mean(rewards)
        mean_best_reward = np.mean(super_rewards)

        print(f"{iteration}. Best individuals:")
        print(np.sort(super_rewards)[::-1])
        print(f"Mean reward: {mean_all_reward}, time: {time.time() - tic}")
        print()
        iteration += 1

    # if i % 20 == 1:  # Write all important info to files every 20 iterations
    #     with open("best_species_pickle_" + str(myRand) + ".txt", "wb") as fp:
    #         pickle.dump(super_actions, fp)
    #     with open("best_species_txt_" + str(myRand) + ".txt", "w") as f:
    #         for item in super_actions:
    #             f.write(str(item))
    #             f.write("\n")
    #     with open("best_species_rewards_" + str(myRand) + ".txt", "w") as f:
    #         for item in super_rewards:
    #             f.write(str(item))
    #             f.write("\n")
    #     with open("best_100_rewards_" + str(myRand) + ".txt", "a") as f:
    #         f.write(str(mean_all_reward) + "\n")
    #     with open("best_elite_rewards_" + str(myRand) + ".txt", "a") as f:
    #         f.write(str(mean_best_reward) + "\n")
    # if i % 200 == 2:  # To create a timeline, like in Figure 3
    #     with open("best_species_timeline_txt_" + str(myRand) + ".txt", "a") as f:
    #         f.write(str(super_actions[0]))
    #         f.write("\n")
