import pickle
from typing import List

import numpy as np

GAMMA_W = 0.15
GAMMA_B = GAMMA_W
BETA = 1.0


class StochNN:
    def __init__(self, inputSize: int, hiddenLayers: List[int], outputSize: int):
        if len(hiddenLayers) == 0:
            raise ValueError("Network must have hidden layer!")

        self.inputSize = inputSize
        self.weights_list: List[np.array] = []
        self.bias_list: List[np.array] = []
        self.nodes_list: List[np.array] = []
        self.avg_list: List[np.array] = []
        self.gamma_w = GAMMA_W
        self.gamma_b = GAMMA_B

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
        self.randomize_weights(0.1)
        self.randomize_biases(0.1)

    def save_model(self, name):
        data = [self.inputSize, self.weights_list, self.bias_list]
        with open(name, 'wb') as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

    def load_model(self, name):
        with open(name, 'rb') as infile:
            result = pickle.load(infile)
            self.inputSize = result[0]
            self.weights_list = result[1]
            self.bias_list = result[2]

    def set_gammas(self, g_w=None, g_b=None):
        if g_w != None:
            self.gamma_w = g_w
        if g_b != None:
            self.gamma_b = g_b

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
        # create vector of random values, uniformly distributed from 0 to 1
        rand = np.random.rand(*input.shape)

        # calculate probability of activation using sigmoid
        val = 1.0 / (1.0 + np.exp(-input))

        # change vector of boolean values to 1 and -1 values
        bl = 2.0 * (val > rand) - 1.0

        # return activation vector and sigmoid for debugging
        return bl, val

    def activation_func_outside(self, input: np.array):
        val = 1.0 / (1.0 + np.exp(-BETA * input))
        return val, val

    def backpropagation(self, desired_output: np.array, average_input: float):
        if desired_output.shape[0] != self.nodes_list[-1].shape[0]:
            raise ValueError("Pattern size error!")
        desired_output_ = desired_output[np.newaxis].T

        def f_prime_inside(index):
            return self.nodes_list[index]
            #return 2.0 * (BETA * self.nodes_list[index] * (1.0 - BETA * self.nodes_list[index]))

        def f_prime_outside(index):
            return (self.nodes_list[index] * (1.0 -  self.nodes_list[index]))

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
            self.weights_list[i] = w - self.gamma_w * derivative

        for i in range(len(self.bias_list)):
            self.bias_list[i] -= self.gamma_b * deltas[i + 1]


if __name__ == "__main__":
    nn = StochNN(2, [32], 1)
    xor_map = [(np.array([1.0, 1.0]), np.array([0.0])),
               (np.array([1.0, 0.0]), np.array([1.0])),
               (np.array([0.0, 1.0]), np.array([1.0])),
               (np.array([0.0, 0.0]), np.array([0.0])),
               ]

    # learning
    learning_curve = []
    MEASURE_POINTS = 40
    # EPOCHS = 6000
    EPOCHS = 1600
    SINGLE_AVERAGE = 20
    for _ in range(EPOCHS):
        average = []
        for _ in range(MEASURE_POINTS):
            for input, output in xor_map:
                # predictions = []
                # for _ in range(SINGLE_AVERAGE):
                #     predictions.append(nn.feed_forward(input))
                # pred_avg = np.array(predictions).mean()
                # nn.backpropagation(output, pred_avg)
                # average.append((pred_avg - output)**2)
                val = nn.feed_forward(input)
                nn.backpropagation(output, val)
                average.append((val - output)**2)
        learning_curve.append(np.array(average).mean())


    for input, output in xor_map:
        ps = []
        for i in range(100):
            ps.append(nn.feed_forward(input))
        npArr = np.array(ps)
        print(input, output, np.mean(npArr), np.std(npArr))

    nn.save_model("test.stoch")

    from matplotlib import pyplot as plt
    plt.title("Accuracy in each epoch")
    plt.xlabel("epochs")
    plt.ylabel("(x-t)^2")
    plt.plot(learning_curve)
    plt.show()
