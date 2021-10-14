from typing import List

import numpy as np

ALPHA_W = 0.0001
BETA = 0.005


class StochNN:
    def __init__(self, inputSize: int, hiddenLayers: List[int], outputSize: int):
        if len(hiddenLayers) == 0:
            raise ValueError("Netowork must have hidden layer!")

        self.inputSize = inputSize
        self.weights_list: List[np.mat] = []
        self.nodes_list: List[np.mat] = []
        self.nodes_bool_list: List[np.mat] = []

        inputWeights = np.zeros((hiddenLayers[0], inputSize))
        self.weights_list.append(inputWeights)
        for i, layerSize in enumerate(hiddenLayers):
            if(i == 0):
                continue
            weights = np.zeros((layerSize, hiddenLayers[i - 1]))
            self.weights_list.append(weights)
        outputWeights = np.zeros((outputSize, hiddenLayers[-1]))
        self.weights_list.append(outputWeights)
        self.randomize_weights()

    def randomize_weights(self, maxRand: float = 1.0) -> None:
        self.weights_list[:] = [maxRand * np.random.rand(
            *w.shape) for w in self.weights_list]

    def feed_forward(self, input: np.array) -> np.array:
        if input.shape[0] != self.inputSize:
            raise ValueError("Input Size error!")

        self.nodes_list = []
        self.nodes_bool_list = []
        self.last_input = np.mat(input)

        middleArr = np.array([])
        for i, w in enumerate(self.weights_list):
            if i == 0:
                middleArr = w @ input
            else:
                middleArr = w @ middleArr
            middleArr, bl = self.activation_func(middleArr)
            self.nodes_list.append(middleArr)
            self.nodes_bool_list.append(bl)
        return middleArr

    def activation_func(self, input: np.array):
        rand = np.random.rand(*input.shape)
        val = 1.0 / (1.0 + np.exp(-2.0 * BETA * input))
        bl = (val > rand)
        return val * bl, bl

    def backpropagation(self, desired_output: np.array):
        if desired_output.shape[0] != self.nodes_list[-1].shape[0]:
            raise ValueError("Patter size error!")

        def f_prime(index):
            return (2.0 * BETA *
                    self.nodes_list[index] * 1.0 - self.nodes_list[index]) * self.nodes_bool_list[index]

        deltas = [None] * (len(self.nodes_list) + 1)
        deltas[-1] = (self.nodes_list[-1] - desired_output) * f_prime(-1)

        for i in range(len(self.nodes_list) - 1, 0, -1):
            deltas[i] = (self.weights_list[i + 1 - 1].transpose()
                         @ deltas[i + 1]) * f_prime(i - 1)

        for i, w in enumerate(self.weights_list):
            derivative = None
            if i == 0:
                derivative = np.mat(
                    deltas[i + 1]).transpose() @ self.last_input
            else:
                derivative = np.mat(
                    deltas[i + 1]).transpose() @ np.mat(self.nodes_list[i - 1])
            self.weights_list[i] = np.array(w - ALPHA_W * derivative)


if __name__ == "__main__":
    nn = StochNN(2, [10, 10], 1)
    xor_map = [(np.array([1.0, 1.0]), np.array([0.0])),
               (np.array([1.0, 0.0]), np.array([1.0])),
               (np.array([0.0, 1.0]), np.array([1.0])),
               (np.array([0.0, 0.0]), np.array([0.0])),
               ]
    # learning
    learning_curve = []
    BATCHES = 40
    EPOCHS = 200
    for _ in range(EPOCHS):
        average = []
        for _ in range(BATCHES):
            for input, output in xor_map:
                prediction = nn.feed_forward(input)
                nn.backpropagation(output)
            if prediction != 0.0:
                average.append((prediction - output)**2)
        learning_curve.append(np.array(average).mean())

    from matplotlib import pyplot as plt
    plt.plot(learning_curve)
    plt.show()
