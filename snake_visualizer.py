

import pygame

class SnakeVisualizer:
    def __init__(self, grid_size, cell_size=20):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.snake_color = (0, 255, 0)
        self.food_color = (255, 0, 0)

    def render(self, snake, food):
        self.screen.fill((0, 0, 0))  # Clear screen with black

        # Draw the snake
        for segment in snake:
            pygame.draw.rect(self.screen, self.snake_color, pygame.Rect(segment[0] * self.cell_size, segment[1] * self.cell_size, self.cell_size, self.cell_size))

        # Draw the food
        pygame.draw.rect(self.screen, self.food_color, pygame.Rect(food[0] * self.cell_size, food[1] * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()  # Update the display

    def tick(self, fps=10):
        self.clock.tick(fps)  # Control game speed (frames per second)

    def quit(self):
        pygame.quit()
