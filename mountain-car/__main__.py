import random
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from typing import Deque

import gym
import numpy as np
from keras import layers, models
from keras.optimizers import Adam


def create_network(learning_rate: float) -> models.Model:
    """Initialise the Neural Network with state as input and Q-value against each action as
    output"""
    model = models.Sequential()
    state_shape = env.observation_space.shape

    model.add(layers.Dense(24, activation="relu", input_shape=state_shape))
    model.add(layers.Dense(48, activation="relu"))
    model.add(layers.Dense(env.action_space.n, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return model


@dataclass
class TrainState:
    """Contains training params and the model"""

    eps_policy = 1.0
    eps_policy_decay = 0.05
    eps_policy_min = 0.01  # policy it at least 1% random
    model: models.Model = field(default_factory=partial(create_network, 0.001))
    replays: Deque = field(default_factory=partial(deque, maxlen=20000))
    n_replay_samples = 32
    discount = 0.99


def get_action(training: TrainState, state: np.array) -> int:
    """Return epsilon-greedy action"""
    epsilon = max(training.eps_policy_min, training.eps_policy)
    if np.random.rand(1) < epsilon:
        action = np.random.randint(0, 3)
    else:
        action = np.argmax(training.model.predict(state.reshape(-1, 2))[0])
    return action


def train(training: TrainState):
    """Update parameters of the NN using Q-learning"""
    if len(training.replays) < training.n_replay_samples:
        return

    replays = random.sample(training.replays, training.n_replay_samples)
    states = np.array([r[0] for r in replays])
    new_states = np.array([r[3] for r in replays])
    targets = training.model.predict(states)
    q_future = training.model.predict(new_states).max(axis=1)

    for i in range(len(targets)):
        _, action, reward, _, done = replays[i]
        targets[i, action] = reward if done else reward + q_future[i] * training.discount
    training.model.fit(states, targets, epochs=1, verbose=0)


def run_episode(training: TrainState, env: gym.core.Env, should_render: bool) -> int:
    """Runs a training episode, i.e. training from the unitialised state of the game"""
    current_state = env.reset()

    done = False
    num_steps = 0
    while not done:
        num_steps += 1
        action = get_action(training, current_state)
        if should_render:
            env.render()

        new_state, reward, done, _ = env.step(action)
        training.replays.append([current_state, action, reward, new_state, done])
        train(training)

        if done:
            break
        current_state = new_state

    training.eps_policy -= training.eps_policy_decay
    return num_steps


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    training = TrainState()
    n_episodes = 400
    for eps_num in range(1, n_episodes + 1):
        num_steps_taken = run_episode(training, env, eps_num % 50 == 0)
        if num_steps_taken == env._max_episode_steps:
            print(f"Failed to finish task in episode {eps_num}")
        else:
            print(f"Success in episode {eps_num}, used {num_steps_taken} steps")
