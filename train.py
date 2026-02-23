from NeuralNetwork import *

model = NeuralNetwork([784, 2048, 2048, 10])

for i in range(len(model.biases)):
    model.biases[i] = np.load(f"models/biases{i}.npy")
    model.weights[i] = np.load(f"models/weights{i}.npy")

imagedata = np.append(np.load("traindata/imagedata0.npy"), np.load("traindata/imagedata1.npy"), axis = 0)
labeldata = np.load("traindata/labeldata.npy")

testimagedata = np.load("testdata/testimagedata.npy")
testlabeldata = np.load("testdata/testlabeldata.npy")

testimagedata = testimagedata.reshape(-1, 784)

imagedata = imagedata.reshape(-1, 784)

epoch = 0
learningrate = 0.1
dropoutprob = 0.5
batchsize = 600
debugbatch = 10000

while True:
    perm = np.random.permutation(60000)
    imagedata = imagedata[perm]
    labeldata = labeldata[perm]

    for i in range(60000 // batchsize):
        wrate, brate = model.backpropogate(imagedata[(i) * batchsize:(i+1) * batchsize].T, labeldata[i * batchsize:(i+1) * batchsize], batchsize, dropoutprob)
        for j in range(len(brate)):
            model.weights[j] -= wrate[j] * learningrate
            model.biases[j] -= brate[j] * learningrate

    epoch += 1
    for i in range(len(model.weights)):
        np.save(f"models/weights{i}", model.weights[i])
        np.save(f"models/biases{i}", model.biases[i])

    error = model.calculateerror(testimagedata[:debugbatch].T, testlabeldata[:debugbatch], debugbatch)
    traincost =  model.calculatecost(imagedata[:debugbatch].T, labeldata[:debugbatch], debugbatch)
    print(f"Epoch: {epoch} | Test set error = {round(error, 4)} | Training set cost = {round(traincost, 4)}")