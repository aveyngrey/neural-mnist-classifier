import pygame
import mnist_classifier

GRID_SIZE = 28
CELL_SIZE = 25
DISPLAY_SIZE = GRID_SIZE * CELL_SIZE

grid_states = [[False for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]

pygame.init()
screen = pygame.display.set_mode((DISPLAY_SIZE, DISPLAY_SIZE))
pygame.display.set_caption("Digit Detector")
clock = pygame.time.Clock()

def draw_grid():
    for x in range(1, GRID_SIZE):
        x *= CELL_SIZE
        pygame.draw.line(screen, pygame.Color("darkgray"), (x, 0), (x, DISPLAY_SIZE))

    for y in range(1, GRID_SIZE):
            y *= CELL_SIZE
            pygame.draw.line(screen, pygame.Color("darkgray"), (0, y), (DISPLAY_SIZE, y))

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid_states[x][y]:
                cell = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, pygame.Color("white"), cell)
 
def main():
    running = True

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


        mouse_buttons = pygame.mouse.get_pressed()
        x, y = pygame.mouse.get_pos()

        if mouse_buttons[0]:
            grid_states[x // CELL_SIZE][y // CELL_SIZE] = True
        if mouse_buttons[2]:
            grid_states[x // CELL_SIZE][y // CELL_SIZE] = False
        
        screen.fill(pygame.Color("black"))
        draw_grid()
        pygame.display.flip()
        clock.tick(45)

    pygame.quit()

if __name__ == "__main__":
    main()