import numpy as np

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1   # learning rate
        self.gamma = 0.9   # discount factor
        self.epsilon = 0.1 # exploration rate

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state):
        key = self.get_state_key(state)

        # Initialize state if not present
        if key not in self.q_table:
            self.q_table[key] = [0, 0]

        # Exploration vs Exploitation
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        else:
            return np.argmax(self.q_table[key])

    def update(self, state, action, reward, next_state):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        # Initialize next state if not present
        if next_key not in self.q_table:
            self.q_table[next_key] = [0, 0]

        # Q-learning formula
        self.q_table[key][action] += self.alpha * (
            reward + self.gamma * max(self.q_table[next_key]) - self.q_table[key][action]
      )
