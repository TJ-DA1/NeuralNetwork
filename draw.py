from scipy.ndimage import gaussian_filter

from NeuralNetwork import *
import pygame, pygame_gui

pygame.init()

model = NeuralNetwork([784, 256, 128, 10])
modeltype = "digit"

for i in range(len(model.biases)):
    model.biases[i] = np.load(f"models/{modeltype}/biases{i}.npy")
    model.weights[i] = np.load(f"models/{modeltype}/weights{i}.npy")

screen = pygame.display.set_mode((504, 504))
manager = pygame_gui.UIManager((504,504))
guessindicator = pygame_gui.elements.UILabel(relative_rect=(0,480,290,20), text = "Null", manager=manager)

grid = np.zeros((28, 28), dtype="int")

while True:
    screen.fill((255, 255, 255))
    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
            raise SystemExit

    if keys[pygame.K_w]:
        x = abs(pygame.mouse.get_pos()[0] - 9) // 18
        y = abs(pygame.mouse.get_pos()[1] - 9) // 18
        grid[y][x] = 255

        grid[min(y + 1, 27)][x] = 255
        grid[y][min(x + 1, 27)] = 255
        grid[min(y + 1, 27)][min(x + 1, 27)] = 255

    elif keys[pygame.K_c]:
        grid = np.zeros((28, 28))

    blurred = gaussian_filter(grid, sigma=0.6)

    for i in range(28):
        for j in range(28):
            col = 255 - blurred[j][i]
            pygame.draw.rect(screen, (col, col, col), (i * 18, j * 18, 18, 18))

    model.calculatelayers(np.reshape(blurred, (784, 1)) / 255)

    guessindicator.set_text(f"Model thinks this is a {np.argmax(model.activations[-1])}, {round(np.max(model.activations[-1] * 100), 2)}% certainty")
    manager.draw_ui(screen)
    manager.update(1/60)

    pygame.display.flip()