from NeuralNetwork import *
import pygame

model = NeuralNetwork([784, 256, 128, 10])
modeltype = "fashion"

for i in range(len(model.biases)):
    model.biases[i] = np.load(f"models/{modeltype}/biases{i}.npy")
    model.weights[i] = np.load(f"models/{modeltype}/weights{i}.npy")

layerspacing = 6000
height = 20 * len(model.activations[0])
screen = pygame.Surface((layerspacing * (len(model.activations) - 1) + 200, height + 200))

screen.fill((0,0,0))

for i in range(len(model.weights)):
    colourarg= np.max(np.abs(model.weights[i]))
    x1 = (i * layerspacing)
    x2 = ((i + 1) * layerspacing)

    for j in range(len(model.weights[i])):
        endpos = (x2 + 100, ((height * (j + 1)) / (len(model.activations[i + 1]) + 1)) + 100)

        for k in range(len(model.weights[i][j])):
            startpos = (x1 + 100, ((height * (k + 1)) / (len(model.activations[i]) + 1)) + 100)

            weight = model.weights[i][j][k]
            normalisedcolour = 155 * ((abs(weight) / colourarg) ** 0.5)
            normalisedcolour += 100
            colour = (0, normalisedcolour, 0) if weight > 0 else (normalisedcolour, 0, 0)

            pygame.draw.line(screen, colour, startpos, endpos, 1 if not i else 3)

for i in range(len(model.activations)):
    colourarg = np.max(np.abs(model.biases[i-1]))
    x = i * layerspacing
    layersize = len(model.activations[i])

    for j in range(layersize):
        y = (height * (j + 1)) / (len(model.activations[i]) + 1)
        if i == 0:
            colour = (255, 255, 255)
            radius = 6

        else:
            bias = model.biases[i-1][j]
            normalisedcolour = 205 * ((abs(bias[0]) / colourarg) ** 0.5)
            normalisedcolour += 50
            colour = (0, normalisedcolour, 0) if bias > 0 else (normalisedcolour, 0, 0)
            radius = 20

        pygame.draw.circle(screen, colour, (x + 100, y + 100), radius)

pygame.image.save(screen, f"imageout/{modeltype}/model.png")