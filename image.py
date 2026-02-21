from NeuralNetwork import *
import pygame

model = NeuralNetwork([784, 40, 20, 16, 10])
for i in range(len(model.biases)):
    model.biases[i] = np.load(f"models/biases{i}.npy")
    model.weights[i] = np.load(f"models/weights{i}.npy")

layerspacing = 4000
height = 20 * len(model.activations[0]) - 10
screen = pygame.Surface((layerspacing * len(model.activations), height + 10))

screen.fill((0,0,0))

for i in range(len(model.weights)):
    colourarg= np.max(np.abs(model.weights[i]))
    x1 = (layerspacing / 2) + (i * layerspacing)
    x2 = (layerspacing * (3/2)) + (i * layerspacing)

    for j in range(len(model.weights[i])):
        endpos = (x2, (height * (j + 1)) / (len(model.activations[i + 1]) + 1))

        for k in range(len(model.weights[i][j])):
            startpos = (x1, (height * (k + 1)) / (len(model.activations[i]) + 1))

            weight = model.weights[i][j][k]
            normalisedcolour = 155 * ((abs(weight) / colourarg) ** 0.5)
            normalisedcolour += 100
            colour = (0, normalisedcolour, 0) if weight > 0 else (normalisedcolour, 0, 0)

            pygame.draw.line(screen, colour, startpos, endpos, 1 if not i else 3)

for i in range(len(model.activations)):
    colourarg = np.max(np.abs(model.biases[i-1]))
    x = (layerspacing / 2) + (i * layerspacing)
    layersize = len(model.activations[i])

    for j in range(layersize):
        y = (height * (j + 1)) / (len(model.activations[i]) + 1)
        if i == 0:
            colour = (255, 255, 255)
            radius = 6

        else:
            bias = model.biases[i-1][j]
            normalisedcolour = 155 * ((abs(bias) / colourarg) ** 0.5)
            normalisedcolour += 100
            colour = (0, normalisedcolour, 0) if bias > 0 else (normalisedcolour, 0, 0)
            radius = 20

        pygame.draw.circle(screen, colour, (x, y), radius)

pygame.image.save(screen, "imageout/model.png")