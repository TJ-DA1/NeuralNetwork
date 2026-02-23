from NeuralNetwork import *

model = NeuralNetwork([784, 2048, 2048, 10])

for i in range(len(model.biases)):
    model.biases[i] = np.load(f"models/biases{i}.npy")
    model.weights[i] = np.load(f"models/weights{i}.npy")

testimagedata = np.load("testdata/testimagedata.npy")
testlabeldata = np.load("testdata/testlabeldata.npy")

testimagedata = testimagedata.reshape(-1, 784, 1)

truthlist = []
cost = 0
for i in random.sample(range(0, 10000), 10000):
    truthlist.append(isworking(model, testimagedata[i], testlabeldata[i]))
    cost += model.calculateerror(testimagedata[i], testlabeldata[i])

    # plt.matshow(testimagedata[i].reshape(28,28,1), cmap="binary")
    # plt.title(f"Model: {np.argmax(model.activations[-1])} Actual: {testlabeldata[i]}")
    # plt.show()

print(f"Prediction rate: {truthlist.count(True)} / {len(truthlist)}")
print(f"Average cost: {cost / 10000}")