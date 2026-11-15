import pygame
import mnist_classifier
import numpy as np
import numpy.typing as npt

def shift_to_center(array: npt.NDArray) -> npt.NDArray:
    GRID_SIZE = len(array)
    
    total_mass = np.sum(array)
    y_indices, x_indices = np.indices(array.shape)
    if total_mass > 0:
        center_x = np.sum(x_indices * array) / total_mass
        center_y = np.sum(y_indices * array) / total_mass
        dx = int(round(center_x - 13.5))
        dy = int(round(center_y - 13.5))

        src_x1 = src_y1 = dst_x1 = dst_y1 = 0
        src_x2 = src_y2 = dst_x2 = dst_y2 = GRID_SIZE - 1


        if dx > 0:
            src_x1 += dx
            dst_x2 -= dx
        elif dx < 0:
            src_x2 += dx
            dst_x1 -= dx

        if dy > 0:
            src_y1 += dy
            dst_y2 -= dy
        elif dy < 0:
            src_y2 += dy
            dst_y1 -= dy

        shifted = np.zeros_like(array)
        shifted[dst_y1:dst_y2, dst_x1:dst_x2] = array[src_y1:src_y2, src_x1:src_x2]
        return shifted
    else:
        return array

def main(weights_hidden, biases_hidden, weights_output, biases_output):
    GRID_SIZE = 28
    CELL_SIZE = 25
    DISPLAY_SIZE = GRID_SIZE * CELL_SIZE

    BLACK = pygame.Color("black")
    WHITE = pygame.Color("white")
    GREEN = pygame.Color("green")
    BRUSH_SIZE = 25

    pygame.init()
    pygame.font.init()
    font = pygame.font.Font(None, 36)

    screen = pygame.display.set_mode((DISPLAY_SIZE, DISPLAY_SIZE))
    pygame.display.set_caption("Interactive Digit Classifier")
    clock = pygame.time.Clock()

    canvas = pygame.Surface((DISPLAY_SIZE, DISPLAY_SIZE))
    canvas.fill(BLACK)

    prediction_text = ""
    last_pos = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.MOUSEBUTTONDOWN:
                prediction_text = ""
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    canvas.fill(BLACK)
                    prediction_text = ""
                if event.key == pygame.K_RETURN:
                    scaled_canvas = pygame.transform.smoothscale(canvas, (GRID_SIZE, GRID_SIZE))
                    array = pygame.surfarray.array3d(scaled_canvas)[:, :, 0]
                    array = array.astype(np.float32) / 255.0
                    array = array.T
                    array = shift_to_center(array)
                    array = array.reshape(1, 784)
                    prediction, confidence = mnist_classifier.predict(
                        array, 
                        weights_hidden, 
                        biases_hidden, 
                        weights_output, 
                        biases_output
                    )        
                    prediction_text = f"Prediction: {prediction} ({confidence:.2%})"
                    print(prediction_text)
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

        if prediction_text:
            text_surface = font.render(prediction_text, True, GREEN)
            text_rect = text_surface.get_rect(topleft=(15, 15))
            background_rect = text_rect.inflate(16, 16)
            pygame.draw.rect(screen, BLACK, background_rect)
            pygame.draw.rect(screen, GREEN, background_rect, 2)
            screen.blit(text_surface, text_rect)
        
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()

if __name__ == "__main__":
    main(0, 0, 0, 0)