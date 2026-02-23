import matplotlib.pyplot as plt

from NeuralNetwork import *
import pygame, pygame_gui

pygame.init()

model = NeuralNetwork([784, 128, 64, 10])
for i in range(len(model.biases)):
    model.biases[i] = np.load(f"models/biases{i}.npy")
    model.weights[i] = np.load(f"models/weights{i}.npy")

screen = pygame.display.set_mode((504, 504))
manager = pygame_gui.UIManager((504,504))
guessindicator = pygame_gui.elements.UILabel(relative_rect=(0,480,290,20), text = "Null", manager=manager)

grid = np.zeros((28, 28), dtype="int")
imagedata = np.load("traindata/drawimagedata.npy")
labeldata = np.load("traindata/drawlabeldata.npy")

while True:
    screen.fill((255, 255, 255))
    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
            raise SystemExit

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                correct = input("Input actual value:\n")
                imagedata = np.append(imagedata, np.reshape(blurred / 255, (1, 28, 28, 1)), axis = 0)
                labeldata = np.append(labeldata, [int(correct)])
                np.save("traindata/drawimagedata.npy", imagedata)
                np.save("traindata/drawlabeldata.npy", labeldata)
                print("New data written")

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