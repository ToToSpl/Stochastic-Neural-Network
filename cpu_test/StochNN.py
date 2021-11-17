from typing import List

import numpy as np

GAMMA_W = 0.5
GAMMA_B = GAMMA_W
BETA = 1.0


class StochNN:
    def __init__(self, inputSize: int, hiddenLayers: List[int], outputSize: int):
        if len(hiddenLayers) == 0:
            raise ValueError("Netowork must have hidden layer!")

        self.inputSize = inputSize
        self.weights_list: List[np.array] = []
        self.bias_list: List[np.array] = []
        self.nodes_list: List[np.array] = []
        self.avg_list: List[np.array] = []

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
        self.randomize_weights(1.0)
        self.randomize_biases(1.0)

    def randomize_weights(self, maxRand: float = 1.0) -> None:
        self.weights_list[:] = [
            maxRand * (2.0 * np.random.rand(*w.shape) - 1.0) for w in self.weights_list]

    def randomize_biases(self, maxRand: float = 1.0) -> None:
        self.bias_list[:] = [
            maxRand * (2.0 * np.random.rand(*b.shape) - 1.0) for b in self.bias_list]

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
            avg = None
            if i == len(self.weights_list)-1:
                middleArr, avg = self.activation_func_outside(middleArr)
            else:
                middleArr, avg = self.activation_func_inside(middleArr)
            self.nodes_list.append(middleArr)
            self.avg_list.append(avg)
        return self.nodes_list[-1]

    def activation_func_inside(self, input: np.array):
        rand = np.random.rand(*input.shape)
        val = 1.0 / (1.0 + np.exp(-BETA * input))
        # val = np.tanh(BETA * input)
        bl = 2.0 * (val > rand) - 1.0
        return bl, val

    def activation_func_outside(self, input: np.array):
        val = 1.0 / (1.0 + np.exp(-BETA * input))
        return val, val

    def backpropagation(self, desired_output: np.array, average_input: float):
        if desired_output.shape[0] != self.nodes_list[-1].shape[0]:
            raise ValueError("Pattern size error!")
        desired_output_ = desired_output[np.newaxis].T

        def f_prime_inside(index):
            return 2.0 * (BETA * self.nodes_list[index] * (1.0 - BETA * self.nodes_list[index]))

        def f_prime_outside(index):
            return (BETA * self.nodes_list[index] * (1.0 - BETA * self.nodes_list[index]))

        deltas = [None] * (len(self.nodes_list) + 1)

        deltas[-1] = (average_input - desired_output_) * \
            f_prime_outside(-1)
        for i in range(len(self.nodes_list) - 1, 0, -1):
            deltas[i] = (self.weights_list[i].T @
                         deltas[i + 1]) * f_prime_inside(i - 1)

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
    nn = StochNN(2, [32], 1)
    xor_map = [(np.array([1.0, 1.0]), np.array([0.0])),
               (np.array([1.0, 0.0]), np.array([1.0])),
               (np.array([0.0, 1.0]), np.array([1.0])),
               (np.array([0.0, 0.0]), np.array([0.0])),
               ]

    # learning
    learning_curve = []
    MEASURE_POINTS = 10
    EPOCHS = 200
    SINGLE_AVERAGE = 20
    for _ in range(EPOCHS):
        average = []
        for _ in range(MEASURE_POINTS):
            for input, output in xor_map:
                predictions = []
                for _ in range(SINGLE_AVERAGE):
                    predictions.append(nn.feed_forward(input))
                pred_avg = np.array(predictions).mean()
                nn.backpropagation(output, pred_avg)
                average.append((pred_avg - output)**2)
        learning_curve.append(np.array(average).mean())
        if learning_curve[-1] < 0.15:
            GAMMA_W = GAMMA_B = 0.005
        if learning_curve[-1] < 0.02:
            break

    for input, output in xor_map:
        ps = [pred_avg]
        for i in range(10):
            ps.append(nn.feed_forward(input))
        print(input, output, np.array(ps).mean())

    from matplotlib import pyplot as plt
    plt.plot(learning_curve)
    plt.show()
