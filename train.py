from NeuralNetwork import *

model = NeuralNetwork([784, 40, 20, 16, 10])

for i in range(len(model.biases)):
    model.biases[i] = np.load(f"models/biases{i}.npy")
    model.weights[i] = np.load(f"models/weights{i}.npy")

imagedata = np.append(np.load("traindata/imagedata0.npy"), np.load("traindata/imagedata1.npy"), axis = 0)
labeldata = np.load("traindata/labeldata.npy")

imagedata = imagedata.reshape(-1, 784)

epoch = 0
learningrate = 0.01
batchsize = 60

while True:
    perm = np.random.permutation(60000)
    imagedata = imagedata[perm]
    labeldata = labeldata[perm]

    for i in range(60000 // batchsize):
        wrate, brate = model.backpropogate(imagedata[(i) * batchsize:(i+1) * batchsize].T, labeldata[i * batchsize:(i+1) * batchsize], batchsize)
        for j in range(len(brate)):
            model.weights[j] -= (wrate[j] / batchsize) * learningrate
            model.biases[j] -= (np.sum(brate[j], axis=1, keepdims=True) / batchsize) * learningrate

    epoch += 1
    for i in range(len(model.weights)):
        np.save(f"models/weights{i}", model.weights[i])
        np.save(f"models/biases{i}", model.biases[i])

    cost = sum([model.calculatecost(imagedata[i].reshape(784,1), labeldata[i]) for i in range(10)]) / 10
    print(f"Epoch: {epoch}, average cost of ten samples = {cost}")