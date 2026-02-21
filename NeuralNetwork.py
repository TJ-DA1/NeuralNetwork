from definitions import *
class NeuralNetwork:
    def __init__(self, structure):
        self.structure = structure
        self.weights = []
        self.biases = []
        self.activations = []

        for i in structure:
            self.activations.append(np.zeros((i, 1)))

        for i in range(len(structure) - 1):
            self.weights.append(np.random.randn(structure[i + 1], structure[i])* np.sqrt(1 / structure[i]))
            self.biases.append(np.random.rand(structure[i + 1], 1))

    def calculatelayers(self, data):
        self.activations[0] = data
        for i in range(len(self.structure) - 2):
            value = self.weights[i] @ self.activations[i] + self.biases[i]
            self.activations[i+1] = leakyrelu(value)

        value = self.weights[-1] @ self.activations[-2] + self.biases[-1]
        self.activations[-1] = sigmoid(value)

    def calculatecost(self, data, expected):
        testset = np.zeros((len(self.activations[-1]), 1))
        testset[expected] = [1]
        self.calculatelayers(data)
        cost = np.sum(np.square(self.activations[-1] - testset)) / 2

        return cost

    def backpropogate(self, data, expected):
        testset = np.zeros((len(self.activations[-1]), 1))
        testset[expected] = [1]
        self.calculatelayers(data)

        activationerror = self.activations[-1] - testset
        errorslist = [None] * (len(self.structure) - 1)
        errorslist[-1] = activationerror

        for i in range(len(errorslist) - 2, -1, -1):
            errorslist[i] = np.multiply((self.weights[i + 1].transpose() @ errorslist[i + 1]), derivativeleakyrelu(self.activations[i + 1]))

        weightrate = []
        for i in range(len(errorslist)):
            weightrate.append(errorslist[i] @ self.activations[i].transpose())

        return weightrate, errorslist