from typing import List

import numpy as np

GAMMA_W = 0.2
GAMMA_B = GAMMA_W


class ClassicNN:
    def __init__(self, inputSize: int, hiddenLayers: List[int], outputSize: int):
        if len(hiddenLayers) == 0:
            raise ValueError("Netowork must have hidden layer!")

        self.inputSize = inputSize
        self.weights_list: List[np.array] = []
        self.bias_list: List[np.array] = []
        self.nodes_list: List[np.array] = []

        inputWeights = np.zeros((hiddenLayers[0], inputSize))
        self.weights_list.append(inputWeights)
        inputBias = np.zeros((hiddenLayers[0], 1))
        self.bias_list.append(inputBias)
        for i, layerSize in enumerate(hiddenLayers):
            if(i == 0):
                continue
            weights = np.zeros((layerSize, hiddenLayers[i - 1]))
            self.weights_list.append(weights)
            bias = np.zeros((layerSize, 1))
            self.bias_list.append(bias)
        outputWeights = np.zeros((outputSize, hiddenLayers[-1]))
        self.weights_list.append(outputWeights)
        outputBias = np.zeros((outputSize, 1))
        self.bias_list.append(outputBias)
        self.randomize_weights()
        self.randomize_biases()

    def randomize_weights(self, maxRand: float = 1.0) -> None:
        self.weights_list[:] = [
            maxRand * np.random.rand(*w.shape) for w in self.weights_list]

    def randomize_biases(self, maxRand: float = 1.0) -> None:
        self.bias_list[:] = [
            maxRand * np.random.rand(*b.shape) for b in self.bias_list]

    def feed_forward(self, input: np.array) -> np.array:
        if input.shape[0] != self.inputSize:
            raise ValueError("Input Size error!")

        self.nodes_list = []
        self.last_input = input[np.newaxis].T

        middleArr = None
        for i, w in enumerate(self.weights_list):
            if i == 0:
                middleArr = w @ self.last_input
            else:
                middleArr = w @ middleArr
            middleArr += self.bias_list[i]
            middleArr = self.activation_func(middleArr)
            self.nodes_list.append(middleArr)
        return self.nodes_list[-1]

    def activation_func(self, input: np.array):
        return 1.0 / (1.0 + np.exp(-input))

    def backpropagation(self, desired_output: np.array):
        if desired_output.shape[0] != self.nodes_list[-1].shape[0]:
            raise ValueError("Pattern size error!")
        desired_output_ = desired_output[np.newaxis].T

        def f_prime(index):
            return self.nodes_list[index] * (1.0 - self.nodes_list[index])

        deltas = [None] * (len(self.nodes_list) + 1)

        deltas[-1] = (self.nodes_list[-1] - desired_output_) * f_prime(-1)
        for i in range(len(self.nodes_list) - 1, 0, -1):
            deltas[i] = (self.weights_list[i].T @
                         deltas[i + 1]) * f_prime(i - 1)

        for i, w in enumerate(self.weights_list):
            derivative = None
            if i == 0:
                derivative = deltas[1] @ self.last_input.T
            else:
                derivative = deltas[i + 1] @ self.nodes_list[i - 1].T
            self.weights_list[i] = w - GAMMA_W * derivative

        for i in range(len(self.bias_list)):
            self.bias_list[i] -= GAMMA_B * deltas[i + 1]


if __name__ == "__main__":
    nn = ClassicNN(2, [5, 2], 1)
    xor_map = [(np.array([1.0, 1.0]), np.array([0.0])),
               (np.array([1.0, 0.0]), np.array([1.0])),
               (np.array([0.0, 1.0]), np.array([1.0])),
               (np.array([0.0, 0.0]), np.array([0.0])),
               ]

    # learning
    learning_curve = []
    BATCHES = 4
    EPOCHS = 2200
    for _ in range(EPOCHS):
        average = []
        for _ in range(BATCHES):
            for input, output in xor_map:
                prediction = nn.feed_forward(input)
                nn.backpropagation(output)
                average.append((prediction - output)**2)
        learning_curve.append(np.array(average).mean())

    for input, output in xor_map:
        p = nn.feed_forward(input)
        print(input, output, p)

    from matplotlib import pyplot as plt

    plt.title("Accuracy in each epoch")
    plt.xlabel("epochs")
    plt.ylabel("(x-t)^2")
    plt.plot(learning_curve)
    plt.show()
