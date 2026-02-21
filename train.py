from NeuralNetwork import *

model = NeuralNetwork([784, 40, 20, 16, 10])
for i in range(len(model.biases)):
    model.biases[i] = np.load(f"models/biases{i}.npy")
    model.weights[i] = np.load(f"models/weights{i}.npy")

imagedata = np.append(np.load("traindata/imagedata0.npy"), np.load("traindata/imagedata1.npy"), axis = 0)
labeldata = np.load("traindata/labeldata.npy")

epoch = 0
learningrate = 0.001
while True:
    epoch += 1
    if epoch % 10000 == 0:
        for i in range(len(model.weights)):
            np.save(f"models/weights{i}", model.weights[i])
            np.save(f"models/biases{i}", model.biases[i])
        print(f"Epoch: {epoch / 10000}")
    indexes = random.sample(range(0, 60000), 60)
    for i in indexes:
        wrate, brate = model.backpropogate(np.reshape(imagedata[i], (784, 1)), labeldata[i])
        for j in range(len(wrate)):
            model.weights[j] -= (wrate[j] / 60) * learningrate
            model.biases[j] -= (brate[j] / 60) * learningrate