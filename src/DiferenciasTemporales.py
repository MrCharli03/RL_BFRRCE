# -*- coding: utf-8 -*-
"""
Author: Esteban Becerra, Carlos Cruzado, Anastasiya Ruzhytska Email: esteban.becerraf@um.es carlos.cruzadoe1@um.es anastasiya.r.r@um.es Date: 2025/02/24
"""

#@title Importación de librerias

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np
import random
from collections import defaultdict

class TabularAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, method="SARSA"):
        """
        Agente tabular que implementa SARSA y Q-Learning.

        Parámetros:
        - env: entorno Gymnasium.
        - alpha: tasa de aprendizaje.
        - gamma: factor de descuento.
        - epsilon: exploración ε-greedy.
        - method: "SARSA" o "Q-Learning".
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.method = method
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))  # Inicializa Q(s,a)

    def policy(self, state):
        """Política ε-greedy."""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Exploración
        return np.argmax(self.Q[state])  # Explotación

    def train(self, num_episodes=5000, epsilon_decay=0.999):
        episode_rewards = []

        for _ in range(num_episodes):
            state = self.env.reset(seed=42)
            action = self.policy(state)
            done = False
            total_reward = 0

            while not done:
                next_state, reward, done, info = self.env.step(action)

                # Definir target antes de actualizar Q
                target = reward  # Por defecto, target = recompensa inmediata

                if self.method == "SARSA":
                    next_action = self.policy(next_state) if not done else None
                    target = reward if done else reward + self.gamma * self.Q[next_state][next_action]

                elif self.method == "Q-Learning":
                    target = reward if done else reward + self.gamma * np.max(self.Q[next_state])

                else:
                    raise ValueError(f"Método '{self.method}' no reconocido. Debe ser 'SARSA' o 'Q-Learning'.")

                self.Q[state][action] += self.alpha * (target - self.Q[state][action])

                state = next_state
                if self.method == "SARSA" and not done:
                    action = next_action
                else:
                    action = self.policy(next_state)

                total_reward += reward

            self.epsilon = max(0.01, self.epsilon * epsilon_decay)  # Se mantiene exploración gradual
            episode_rewards.append(total_reward)

        return episode_rewards
