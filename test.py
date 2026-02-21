from NeuralNetwork import *

model = NeuralNetwork([784, 40, 20, 16, 10])
for i in range(len(model.biases)):
    model.biases[i] = np.load(f"models/biases{i}.npy")
    model.weights[i] = np.load(f"models/weights{i}.npy")

testimagedata = np.load("testdata/testimagedata.npy")
testlabeldata = np.load("testdata/testlabeldata.npy")

truthlist = []
cost = 0
for i in random.sample(range(0, 9999), 9999):
    truthlist.append(isworking(model, np.reshape(testimagedata[i], (784, 1)), testlabeldata[i]))
    cost += model.calculatecost(np.reshape(testimagedata[i], (784, 1)), testlabeldata[i])

    plt.matshow(testimagedata[i], cmap="binary")
    plt.title(f"Model: {np.argmax(model.activations[-1])} Actual: {testlabeldata[i]}")
    plt.show()

print(f"Prediction rate: {truthlist.count(True)} / {len(truthlist)}")
print(f"Average cost: {cost / 9999}")