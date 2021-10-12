from typing import List

import numpy as np

BETA = 1.0


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

    def randomize_weights(self, maxRand: float = 1.0):
        self.weights_list[:] = [maxRand * np.random.rand(
            *w.shape) for w in self.weights_list]

    def feed_forward(self, input: np.array):
        if(input.shape[0] != self.inputSize):
            raise ValueError("Input Size error!")

        self.nodes_list = []
        self.nodes_bool_list = []

        middleArr = np.array([])
        for i, w in enumerate(self.weights_list):
            if i == 0:
                middleArr = np.matmul(w, input)
            else:
                middleArr = np.matmul(w, middleArr)
            middleArr, bl = self.activation_func(middleArr)
            self.nodes_list.append(middleArr)
            self.nodes_bool_list.append(bl)
        return middleArr

    def activation_func(self, input: np.array):
        rand = np.random.rand(*input.shape)
        val = 1.0 / (1.0 + np.exp(-2.0 * BETA * input))
        bl = (val > rand)
        return val * bl, bl


if __name__ == "__main__":
    nn = StochNN(2, [3, 4], 1)
    nn.feed_forward(np.array([1, 1]))
