import numpy as np
import random


class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        data_collector,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.data_collector = data_collector  # For logging and learning from data
        self.model = None  # This will be set after training

    def choose_action(self, state):
        if self.model:
            # Use the model to predict risky actions (low reward or high death likelihood)
            death_risks = [
                self.model.predict(np.array(state + [action]).reshape(1, -1))
                for action in range(self.action_size)
            ]
            safest_action = np.argmin(
                death_risks
            )  # Choose the action with the lowest risk
            return safest_action

        # Otherwise, use Q-learning exploration/exploitation
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1, 2, 3])  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state, done):
        # Log the data for the snake’s current state, action, and result
        self.data_collector.log_data(state, action, reward)

        # Standard Q-learning update
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (
            reward
            + self.gamma * self.q_table[next_state][best_next_action]
            - self.q_table[state][action]
        )

        if done:
            # Train the model after each snake dies
            self.model = self.data_collector.train_model()

        self.epsilon *= self.epsilon_decay  # Decay the exploration rate over time
