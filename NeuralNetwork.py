from definitions import *
class NeuralNetwork:
    def __init__(self, structure, reluparam = 0.01, l2lambda = 10e-3):
        self.structure = structure
        self.weights = []
        self.biases = []
        self.activations = []
        self.unactivated = []

        self.reluparam = reluparam
        self.l2lambda = l2lambda

        for i in structure:
            self.activations.append(np.zeros((i, 1)))
            self.unactivated.append(np.zeros((i, 1)))

        for i in range(len(structure) - 1):
            self.weights.append(np.random.randn(structure[i + 1], structure[i]) * np.sqrt(2 / structure[i]))
            self.biases.append(np.zeros((structure[i + 1], 1)))

    def calculatelayers(self, data, dropoutprob = 0.2):
        self.activations[0] = data
        for i in range(len(self.structure) - 2):
            value = self.weights[i] @ self.activations[i] + self.biases[i]
            self.unactivated[i+1] = value
            self.activations[i+1] = leakyrelu(value, self.reluparam)

        value = self.weights[-1] @ self.activations[-2] + self.biases[-1]
        self.unactivated[-1] = value
        self.activations[-1] = softmax(value)

        if dropoutprob:
            for i in range(len(self.activations) - 2):
                dropouts = np.where(np.random.rand(*self.activations[i+1].shape) > dropoutprob, 1, 0)
                self.activations[i+1] *= (dropouts / (1 - dropoutprob))


    def calculatecost(self, data, expected, batches = 1):
        testset = np.zeros((self.structure[-1], batches))
        testset[expected, np.arange(batches)] = 1
        self.calculatelayers(data)
        cost = -np.sum(testset * np.log(self.activations[-1] + 1e-9))

        cost += (self.l2lambda / 2) * sum(np.sum(np.square(self.weights[i])) for i in range(len(self.weights)))

        return cost / batches

    def calculateerror(self, data, expected, batches = 1):
        testset = np.zeros((self.structure[-1], batches))
        testset[expected, np.arange(batches)] = 1
        self.calculatelayers(data)
        error = np.sum(np.square(self.activations[-1] - testset)) / 2

        return error / batches

    def backpropogate(self, data, expected, batches, dropoutprob = 0):
        testset = np.zeros((self.structure[-1], batches))
        testset[expected, np.arange(batches)] = 1
        self.calculatelayers(data, dropoutprob)

        activationerror = self.activations[-1] - testset
        errorslist = [None] * (len(self.structure) - 1)
        errorslist[-1] = activationerror

        for i in range(len(errorslist) - 2, -1, -1):
            errorslist[i] = np.multiply((self.weights[i + 1].transpose() @ errorslist[i + 1]), derivativeleakyrelu(self.unactivated[i + 1], self.reluparam))

        weightrate = []
        biasrate = []
        for i in range(len(errorslist)):
            weightrate.append((errorslist[i] @ self.activations[i].transpose()) / batches)
            biasrate.append(np.sum(errorslist[i], axis=1, keepdims = True) / batches)

        weightrate = [weightrate[i] + ((self.l2lambda / batches) * self.weights[i]) for i in range(len(weightrate))]

        return weightrate, biasrate