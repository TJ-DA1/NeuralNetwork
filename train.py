from NeuralNetwork import *

model = NeuralNetwork([784, 256, 128, 10])
modeltype = "fashion"

for i in range(len(model.biases)):
    model.biases[i] = np.load(f"models/{modeltype}/biases{i}.npy")
    model.weights[i] = np.load(f"models/{modeltype}/weights{i}.npy")

imagedata = np.append(np.load(f"traindata/{modeltype}/imagedata0.npy"), np.load(f"traindata/{modeltype}/imagedata1.npy"), axis = 0)
labeldata = np.load(f"traindata/{modeltype}/labeldata.npy")

testimagedata = np.load(f"testdata/{modeltype}/testimagedata.npy")
testlabeldata = np.load(f"testdata/{modeltype}/testlabeldata.npy")

imagedata = imagedata.reshape(-1, 784)
testimagedata = testimagedata.reshape(-1, 784)

epochs = 100

epoch = 0
learningrate = 0.1
dropoutprob = 0.2
batchsize = 60
debugbatch = 10000

for i in range(epochs):
    perm = np.random.permutation(len(imagedata))
    imagedata = imagedata[perm]
    labeldata = labeldata[perm]

    for i in range(int(len(imagedata) / batchsize)):
        wrate, brate = model.backpropogate(imagedata[(i) * batchsize:(i+1) * batchsize].T, labeldata[i * batchsize:(i+1) * batchsize], batchsize, dropoutprob)
        for j in range(len(brate)):
            model.weights[j] -= wrate[j] * learningrate
            model.biases[j] -= brate[j] * learningrate

    epoch += 1
    learningrate *= 0.95 if epoch > 30 else 1

    for i in range(len(model.weights)):
        np.save(f"models/{modeltype}/weights{i}", model.weights[i])
        np.save(f"models/{modeltype}/biases{i}", model.biases[i])

    error = model.calculateerror(testimagedata[:debugbatch].T, testlabeldata[:debugbatch], debugbatch)
    traincost =  model.calculatecost(imagedata[:debugbatch].T, labeldata[:debugbatch], debugbatch)
    print(f"Epoch: {epoch}\t|\tTest set error: {round(error[0], 4)}\t|\tPrediction rate: {np.unique(np.where(error[1] == testlabeldata[:len(error[1])], 1, 0), return_counts=True)[1][1]} / {len(error[1])} \t|\tTraining set cost: {round(traincost, 4)}")