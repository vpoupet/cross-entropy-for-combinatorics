import math
import pickle
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import tensorflow

N = 19  # number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm
# The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)
MYN = int(N * (N - 1) / 2)

# Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
LEARNING_RATE = 0.0001
BATCH_SIZE = 1000  # number of new sessions per iteration
PERCENTILE = 93  # top 100-X percentile we are learning from
SUPER_PERCENTILE = 94  # top 100-X percentile that survives to next iteration

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
GAME_LENGTH = MYN

INF = 100


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


def nb_components(g):
    laplacian = scipy.sparse.csgraph.laplacian(g)
    return np.linalg.eigvals(laplacian).count(0)


def calcScore(state):
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
    myScore = math.sqrt(N - 1) + 1 - lambda1 - mu
    if myScore > 0:
        # You have found a counterexample. Do something with it.
        print(state)
        nx.draw_kamada_kawai(g)
        plt.show()
        exit()

    return myScore


# No need to change anything below here.


def generate_session(model, batch_size):
    states = np.zeros((batch_size, GAME_LENGTH, STATE_SIZE), dtype=int)
    actions = np.zeros([batch_size, GAME_LENGTH], dtype=int)
    prob = np.zeros(batch_size)
    step_i = 0
    step_j = 1
    score = np.zeros((batch_size,))

    for step in range(GAME_LENGTH):
        states[:, step, MYN + step_i] = 1
        states[:, step, MYN + step_j] = 1

        prob = model.predict(states[:, step, :], batch_size=batch_size)

        actions[:, step] = np.random.random(size=(batch_size,)) < prob[:, 0]
        states[:, step:, step] = np.repeat(
            actions[:, step, np.newaxis], GAME_LENGTH - step, axis=1
        )

        step_j += 1
        if step_j >= N:
            step_i += 1
            step_j = step_i + 1

    for i in range(batch_size):
        score[i] = calcScore(states[i, GAME_LENGTH - 1, :])

    return states, actions, score


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    This function was mostly taken from https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
    If this function is the bottleneck, it can easily be sped up using numba
    """
    counter = BATCH_SIZE * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = []
    elite_actions = []
    elite_rewards = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold - 0.0000001:
            if (counter > 0) or (rewards_batch[i] >= reward_threshold + 0.0000001):
                for item in states_batch[i]:
                    elite_states.append(item.tolist())
                for item in actions_batch[i]:
                    elite_actions.append(item)
            counter -= 1
    elite_states = np.array(elite_states, dtype=int)
    elite_actions = np.array(elite_actions, dtype=int)
    return elite_states, elite_actions


def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
    """
    Select all the sessions that will survive to the next generation
    Similar to select_elites function
    If this function is the bottleneck, it can easily be sped up using numba
    """

    counter = BATCH_SIZE * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch, percentile)

    super_states = []
    super_actions = []
    super_rewards = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold - 0.0000001:
            if (counter > 0) or (rewards_batch[i] >= reward_threshold + 0.0000001):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
                counter -= 1
    super_states = np.array(super_states, dtype=int)
    super_actions = np.array(super_actions, dtype=int)
    super_rewards = np.array(super_rewards)
    return super_states, super_actions, super_rewards


super_states = np.empty((0, GAME_LENGTH, STATE_SIZE), dtype=int)
super_actions = np.array([], dtype=int)
super_rewards = np.array([])
sessgen_time = 0
fit_time = 0
score_time = 0


myRand = random.randint(0, 1000)  # used in the filename

for i in range(1000000):  # 1000000 generations should be plenty
    # generate new sessions
    # performance can be improved with joblib
    tic = time.time()
    # change 0 to 1 to print out how much time each step in generate_session takes
    sessions = generate_session(model, BATCH_SIZE)

    states_batch = np.array(sessions[0], dtype=int)
    actions_batch = np.array(sessions[1], dtype=int)
    rewards_batch = np.array(sessions[2])
    # states_batch = np.transpose(states_batch, axes=[0, 2, 1])

    states_batch = np.append(states_batch, super_states, axis=0)

    if i > 0:
        actions_batch = np.append(actions_batch, np.array(super_actions), axis=0)
    rewards_batch = np.append(rewards_batch, super_rewards)

    elite_states, elite_actions = select_elites(
        states_batch, actions_batch, rewards_batch, percentile=PERCENTILE
    )  # pick the sessions to learn from

    super_sessions = select_super_sessions(
        states_batch, actions_batch, rewards_batch, percentile=SUPER_PERCENTILE
    )  # pick the sessions to survive

    super_sessions = [
        (super_sessions[0][i], super_sessions[1][i], super_sessions[2][i])
        for i in range(len(super_sessions[2]))
    ]
    super_sessions.sort(key=lambda super_sessions: super_sessions[2], reverse=True)
    select3_time = time.time() - tic

    model.fit(elite_states, elite_actions)  # learn from the elite sessions

    super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
    super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
    super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]

    rewards_batch.sort()
    mean_all_reward = np.mean(rewards_batch[-100:])
    mean_best_reward = np.mean(super_rewards)

    print("\n" + str(i) + ". Best individuals: " + str(np.flip(np.sort(super_rewards))))

    # uncomment below line to print out how much time each step in this loop takes.
    print(f"Mean reward: {mean_all_reward}, time: {time.time() - tic}")

    if i % 20 == 1:  # Write all important info to files every 20 iterations
        with open("best_species_pickle_" + str(myRand) + ".txt", "wb") as fp:
            pickle.dump(super_actions, fp)
        with open("best_species_txt_" + str(myRand) + ".txt", "w") as f:
            for item in super_actions:
                f.write(str(item))
                f.write("\n")
        with open("best_species_rewards_" + str(myRand) + ".txt", "w") as f:
            for item in super_rewards:
                f.write(str(item))
                f.write("\n")
        with open("best_100_rewards_" + str(myRand) + ".txt", "a") as f:
            f.write(str(mean_all_reward) + "\n")
        with open("best_elite_rewards_" + str(myRand) + ".txt", "a") as f:
            f.write(str(mean_best_reward) + "\n")
    if i % 200 == 2:  # To create a timeline, like in Figure 3
        with open("best_species_timeline_txt_" + str(myRand) + ".txt", "a") as f:
            f.write(str(super_actions[0]))
            f.write("\n")
