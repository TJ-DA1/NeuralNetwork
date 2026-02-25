from NeuralNetwork import *
from matplotlib import pyplot as plt

model = NeuralNetwork([784, 256, 128, 10])
modeltype = "digit"

for i in range(len(model.biases)):
    model.biases[i] = np.load(f"models/{modeltype}/biases{i}.npy")
    model.weights[i] = np.load(f"models/{modeltype}/weights{i}.npy")

testimagedata = np.load(f"testdata/{modeltype}/testimagedata.npy")
testlabeldata = np.load(f"testdata/{modeltype}/testlabeldata.npy")

testimagedata = testimagedata.reshape(-1, 784)

print(testimagedata.dtype)

cost, guessvalue = model.calculateerror(testimagedata.T, testlabeldata, 10000)
truthlist = np.where(guessvalue == testlabeldata, 1, 0)

for i in range(100):
    plt.matshow(testimagedata[i].reshape(28,28,1), cmap="binary")
    plt.title(f"Model: {guessvalue[i]} Actual: {testlabeldata[i]}")
    plt.show()

print(f"Prediction rate: {np.unique(truthlist, return_counts=True)[1][1]} / {len(truthlist)}")
print(f"Average cost: {cost}")