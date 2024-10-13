import random
import numpy as np

class SnakeGame:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [
            (self.grid_size // 2, self.grid_size // 2)
        ]  # Snake starts at the center
        self.direction = (0, 1)  # Initial direction: right
        self.food = self._place_food()
        self.done = False
        self.score = 0
        self.hp = 100  # Snake starts with 100 HP
        return self._get_state()

    def _place_food(self):
        while True:
            food = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if food not in self.snake:
                return food

    def step(self, action):
        # Map actions to direction changes
        if action == 0:  # up
            self.direction = (-1, 0)
        elif action == 1:  # down
            self.direction = (1, 0)
        elif action == 2:  # left
            self.direction = (0, -1)
        elif action == 3:  # right
            self.direction = (0, 1)

        # Move the snake
        new_head = (
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1],
        )
        self.snake = [new_head] + self.snake[:-1]

        # Check for collisions
        if (
            new_head in self.snake[1:]  # Snake collides with itself
            or new_head[0] < 0 or new_head[1] < 0  # Snake hits walls
            or new_head[0] >= self.grid_size or new_head[1] >= self.grid_size
        ):
            self.done = True
            return self._get_state(), -1, self.done  # Reward = -1 (collision)

        # Check for food
        if new_head == self.food:
            self.snake.append(self.snake[-1])  # Grow the snake
            self.food = self._place_food()
            self.score += 1
            self.hp = 100  # Reset HP when snake eats food
            return self._get_state(), 1, False  # Reward = 1 (food eaten)

        # If the snake moves without eating the food, lose 1 HP
        self.hp -= 1
        if self.hp <= 0:
            self.done = True
            return self._get_state(), -1, self.done  # Lose when HP reaches 0

        return self._get_state(), 0, False  # No reward, game continues

    def _get_state(self):
        # Flatten the snake positions (x, y) pairs into a single list
        snake_positions = [coord for segment in self.snake for coord in segment]
        
        # Food and direction are already in a simple format, we can append them directly
        state = snake_positions + list(self.food) + list(self.direction) + [self.hp]

        # Convert the state to a NumPy array
        return np.array(state)


