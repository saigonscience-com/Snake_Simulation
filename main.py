from game import SnakeGame
from q_learning_agent import QLearningAgent
from data_collector import DataCollector

# Configuration values (moved from config.py)
GRID_SIZE = 20  # Size of the game grid
EPISODES = 10000  # Number of episodes for training


def main():
    data_collector = DataCollector()
    game = SnakeGame(grid_size=GRID_SIZE)
    agent = QLearningAgent(state_size=100, action_size=4, data_collector=data_collector)

    for episode in range(EPISODES):  # Train for a set number of episodes
        state = game.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = game.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state

        print(f"Episode {episode}, Score: {game.score}")


if __name__ == "__main__":
    main()
