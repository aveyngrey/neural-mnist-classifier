import pygame
import mnist_classifier
import numpy as np

def draw_grid(screen, grid_states):
    GRID_SIZE = 28
    CELL_SIZE = 25
    DISPLAY_SIZE = GRID_SIZE * CELL_SIZE

    for x in range(1, GRID_SIZE):
        x *= CELL_SIZE
        pygame.draw.line(screen, pygame.Color("darkgray"), (x, 0), (x, DISPLAY_SIZE))

    for y in range(1, GRID_SIZE):
            y *= CELL_SIZE
            pygame.draw.line(screen, pygame.Color("darkgray"), (0, y), (DISPLAY_SIZE, y))

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid_states[y][x]:
                cell = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, pygame.Color("white"), cell)
 
def main(weights_hidden, biases_hidden, weights_output, biases_output):
    running = True

    GRID_SIZE = 28
    CELL_SIZE = 25
    DISPLAY_SIZE = GRID_SIZE * CELL_SIZE

    grid_states = [[False for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]

    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_SIZE, DISPLAY_SIZE))
    pygame.display.set_caption("Interactive Digit Classifier")
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    for x in range(GRID_SIZE):
                        for y in range(GRID_SIZE):
                            grid_states[x][y] = False
                if event.key == pygame.K_RETURN:
                    np_array_grid = np.array(grid_states)
                    np_array_grid = np_array_grid.astype(int)
                    np_array_grid = np.ravel(np_array_grid)
                    np_array_grid = np.reshape(np_array_grid, (1, 784))
                    print(mnist_classifier.predict(np_array_grid, weights_hidden, biases_hidden, weights_output, biases_output))


        mouse_buttons = pygame.mouse.get_pressed()
        x, y = pygame.mouse.get_pos()

        directions = ((0, 1), (1, 0), (0, -1), (-1, 0))

        if 0 <= x <= DISPLAY_SIZE and 0 <= y <= DISPLAY_SIZE:
            if mouse_buttons[0]:
                grid_states[y // CELL_SIZE][x // CELL_SIZE] = True
                for direction in directions:
                    new_x = direction[0] + x // CELL_SIZE
                    new_y = direction[1] + y // CELL_SIZE
                    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                        grid_states[new_y][new_x] = True

            if mouse_buttons[2]:
                grid_states[y // CELL_SIZE][x // CELL_SIZE] = False
                for direction in directions:
                    new_x = direction[0] + x // CELL_SIZE
                    new_y = direction[1] + y // CELL_SIZE
                    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                        grid_states[new_y][new_x] = False
            
        
        screen.fill(pygame.Color("black"))
        draw_grid(screen, grid_states)
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()

if __name__ == "__main__":
    main(0, 0, 0, 0)