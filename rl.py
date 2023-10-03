from contextlib import suppress

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import stable_baselines3 as sb3
import sb3_contrib
from stable_baselines3.common import env_checker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.ppo import MlpPolicy

import rewards


class CustomCallback(BaseCallback):
    def __init__(self, model_name):
        super(CustomCallback, self).__init__()
        self.model_name = model_name

    def _on_training_start(self) -> None:
        pass

    def _on_step(self):
        if self.n_calls % 100000 == 0:
            self.model.save(self.model_name)
            print("Model saved")


class GraphEnv(gym.Env):
    def __init__(self, reward, nb_vertices=20, model_name="model"):
        self.reward = reward
        self.nb_vertices = nb_vertices
        self.nb_edges = nb_vertices * (nb_vertices - 1) // 2
        self.graph = np.zeros(self.nb_edges, dtype=np.int8)
        self.position = 0
        self.i = 0
        self.j = 1
        self.observation_space = spaces.MultiBinary(self.nb_edges + self.nb_vertices)
        self.action_space = spaces.Discrete(2)
        self.model_name = model_name
        self.make_model()
        self.partial_reward = 0

    def make_model(self):
        model_class = sb3.DQN
        try:
            self.model = model_class.load(self.model_name, env=self)
        except FileNotFoundError:
            self.model = model_class("MlpPolicy", self, verbose=1)

    def get_observation(self):
        obs = np.append(self.graph, np.zeros(self.nb_vertices, dtype=np.int8))
        with suppress(IndexError):
            obs[self.nb_edges + self.i] = 1
            obs[self.nb_edges + self.j] = 1
        return obs

    def get_info(self):
        return {}

    def get_terminated(self):
        return self.position >= self.nb_edges

    def get_reward(self):
        if self.get_terminated():
            return self.reward(self.graph, self.nb_vertices)
        return 0
        # reward = self.reward(self.graph, self.nb_vertices)
        # step_reward = reward - self.partial_reward
        # self.partial_reward = reward
        # return step_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.graph = np.zeros(self.nb_edges, dtype=np.int8)
        self.position = 0
        self.i = 0
        self.j = 1
        self.partial_reward = 0
        return self.get_observation(), self.get_info()

    def step(self, action):
        self.graph[self.position] = action
        self.position += 1
        self.j += 1
        if self.j >= self.nb_vertices:
            self.i += 1
            self.j = self.i + 1
        return (
            self.get_observation(),
            self.get_reward(),
            self.get_terminated(),
            False,
            self.get_info(),
        )

    def render(self):
        pass

    def close(self):
        pass

    def check(self):
        self.reset()
        return env_checker.check_env(self, warn=True)

    def play(self, nb_plays=1):
        results = []
        for _ in range(nb_plays):
            obs, _ = self.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.step(action)
            results.append((self.graph, reward))
        return results

    def learn(self, total_timesteps=1e5):
        self.model.learn(
            total_timesteps=total_timesteps, callback=CustomCallback(self.model_name)
        )

        self.model.save(self.model_name)
        mean_reward, std = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10)
        print(f"Mean reward: {mean_reward}, Std. deviation: {std}")


def main():
    env = GraphEnv(
        reward=rewards.get_reward_randic_radius,
        nb_vertices=12,
        model_name="model_randic_DQN_12",
    )
    # env.learn(total_timesteps=1e7)
    results = env.play(100)
    i = np.argmax([r for _, r in results])
    print(results[i])


if __name__ == "__main__":
    # check_env()
    main()
