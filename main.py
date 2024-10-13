from game import SnakeGame
from q_learning_agent import QLearningAgent
from data_collector import DataCollector
from snake_visualizer import SnakeVisualizer
from data_logger import DataLogger

GRID_SIZE = 20
EPISODES = 10000

def main():
    data_collector = DataCollector()  # Collects game data
    game = SnakeGame(grid_size=GRID_SIZE)
    agent = QLearningAgent(state_size=100, action_size=4, data_collector=data_collector, grid_size=GRID_SIZE)
    visualizer = SnakeVisualizer(GRID_SIZE)  # Visualizer to display the game
    logger = DataLogger()  # Logger to save the training data

    for episode in range(EPISODES):
        state = game.reset()
        done = False

        while not done:
            action = agent.choose_action(state)  # Use the improved greedy strategy to choose action
            next_state, reward, done = game.step(action)

            # Render the game
            visualizer.render(game.snake, game.food)
            visualizer.tick()

            agent.learn(state, action, reward, next_state, done)  # Learn from each step
            state = next_state

        logger.log(episode, game.score, agent.epsilon)  # Log episode data
        print(f"Episode {episode}, Score: {game.score}")

    logger.save()  # Save the data after training
    visualizer.quit()

if __name__ == "__main__":
    main()
