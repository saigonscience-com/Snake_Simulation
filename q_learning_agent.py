import numpy as np
import random
class QLearningAgent:
    def __init__(self, state_size, action_size, data_collector, grid_size, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}  # Use a dictionary instead of a fixed-size array
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.grid_size = grid_size  # Store grid size to prevent hitting walls
        self.data_collector = data_collector  # For logging and learning from data
        self.model = None  # This will be set after training

    def choose_action(self, state):
        state_tuple = tuple(state)  # Convert the state to a hashable tuple

        # If we're using epsilon-greedy, we sometimes explore
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1, 2, 3])  # Random action (explore)

        # Otherwise, exploit the best known action from Q-learning
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_tuple])  # Exploit the best action

    def learn(self, state, action, reward, next_state, done):
        """
        Updates the Q-value for the given state-action pair using the Q-learning update rule.
        """
        state_tuple = tuple(state)  # Convert the state to a hashable tuple
        next_state_tuple = tuple(next_state)  # Convert next_state to a hashable tuple

        # Initialize Q-values for new states
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_size)
        if next_state_tuple not in self.q_table:
            self.q_table[next_state_tuple] = np.zeros(self.action_size)

        # Find the best action for the next state
        best_next_action = np.argmax(self.q_table[next_state_tuple])

        # Q-learning formula to update the Q-value of the current state-action pair
        self.q_table[state_tuple][action] = self.q_table[state_tuple][action] + self.alpha * (
            reward
            + self.gamma * self.q_table[next_state_tuple][best_next_action]
            - self.q_table[state_tuple][action]
        )

        # Decay the exploration rate over time
        if done:
            self.epsilon *= self.epsilon_decay

        # Train the model at the end of the episode
        if done:
            self.model = self.data_collector.train_model()







