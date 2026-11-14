import pygame
import mnist_classifier
import numpy as np
 
def main(weights_hidden, biases_hidden, weights_output, biases_output):
    GRID_SIZE = 28
    CELL_SIZE = 25
    DISPLAY_SIZE = GRID_SIZE * CELL_SIZE

    BLACK = pygame.Color("black")
    WHITE = pygame.Color("white")
    BRUSH_SIZE = 25

    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_SIZE, DISPLAY_SIZE))
    pygame.display.set_caption("Interactive Digit Classifier")
    clock = pygame.time.Clock()

    canvas = pygame.Surface((DISPLAY_SIZE, DISPLAY_SIZE))
    canvas.fill(BLACK)

    last_pos = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    canvas.fill(BLACK)
                if event.key == pygame.K_RETURN:
                    scaled_canvas = pygame.transform.smoothscale(canvas, (GRID_SIZE, GRID_SIZE))
                    array = pygame.surfarray.array3d(scaled_canvas)[:, :, 0]
                    array = array.astype(np.float32) / 255.0
                    array = array.T
                    array = array.reshape(1, 784)
                    prediction, confidence = mnist_classifier.predict(array, 
                                                                      weights_hidden=weights_hidden,
                                                                      biases_hidden=biases_hidden,
                                                                      weights_output=weights_output,
                                                                      biases_output=biases_output)
                    print(f"Prediction: {prediction}")
                    print(f"Confidence: {confidence:.2%}")
                    print("-------------------------------")


        mouse_buttons = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        mouse_x, mouse_y = mouse_pos

        if mouse_buttons[0]:
            pygame.draw.circle(canvas, WHITE, mouse_pos, BRUSH_SIZE - 1)
            if last_pos is not None:
                pygame.draw.line(canvas, WHITE, mouse_pos, last_pos, width=BRUSH_SIZE * 2)
            last_pos = mouse_pos
        else:
            last_pos = None
        
        screen.blit(canvas, (0, 0))
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()

if __name__ == "__main__":
    main(0, 0, 0, 0)