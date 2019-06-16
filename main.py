import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def dSigmoid(x):
    return x * (1 - x)

class Network:
    def __init__(self, x, y):
        self.x = x
        self.weights1 = np.random.rand(self.x.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.x, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        weights2_change = np.dot(self.layer1.T, (2 * (self.y - self.output)) * dSigmoid(self.output))
        weights1_change = np.dot(self.x.T, (np.dot((2 * (self.y - self.output) * dSigmoid(self.output)), self.weights2.T) * dSigmoid(self.layer1)))

        self.weights2 += weights2_change
        self.weights1 += weights1_change

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
Y = np.array([[0], [1], [1], [0]])
network = Network(X, Y)

for i in range(1000):
    network.feedforward()
    network.backprop()

print(network.output)
