import pickle
from typing import List

import numpy as np
import cupy as cp

GAMMA_W = 0.2
GAMMA_B = GAMMA_W
BETA = 1.0
RANDOM_START_RANGE = 0.1


class StochNN_GPU:
    def __init__(self, inputSize: int, hiddenLayers: List[int], outputSize: int):
        if len(hiddenLayers) == 0:
            raise ValueError("Netowork must have hidden layer!")

        self.inputSize = inputSize
        self.weights_list: List[cp.array] = []
        self.bias_list: List[cp.array] = []
        self.nodes_list: List[cp.array] = []
        self.gamma_w = GAMMA_W
        self.gamma_b = GAMMA_B

        self.weights_list.append(self.__random_matrix(
            RANDOM_START_RANGE, (hiddenLayers[0], inputSize)))
        self.bias_list.append(self.__random_matrix(
            RANDOM_START_RANGE, (hiddenLayers[0], 1)))
        for i, layerSize in enumerate(hiddenLayers):
            if(i == 0):
                continue
            self.weights_list.append(self.__random_matrix(
                RANDOM_START_RANGE, (layerSize, hiddenLayers[i - 1])))
            self.bias_list.append(self.__random_matrix(
                RANDOM_START_RANGE, (layerSize, 1)))

        self.weights_list.append(self.__random_matrix(
            RANDOM_START_RANGE, (outputSize, hiddenLayers[-1])))
        self.bias_list.append(self.__random_matrix(
            RANDOM_START_RANGE, (outputSize, 1)))

        # __ KERNELS __
        self.activation_func_inside = cp.ElementwiseKernel(
            "float32 x, float32 rand",
            "float32 z",
            """
            float val = 1.0 / (1.0 + expf(-x));
            if(val > rand)
                z = 1.0;
            else
                z = -1.0;
            """,
            "activation_func_inside")

        self.activation_func_outside = cp.ElementwiseKernel(
            "float32 x",
            "float32 y",
            """
            y = 1.0 / (1.0 / expf(-x));
            """,
            "activation_func_outside")

    def __random_matrix(self, range, size):
        arr = cp.random.rand(*size, dtype=cp.float32)
        arr = range * (2.0 * arr - 1.0)
        return arr

    def save_model(self, name):
        data = [self.inputSize, self.weights_list, self.bias_list]
        with open(name, 'wb') as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

    def load_model(self, name):
        with open(name, 'rb') as infile:
            result = pickle.load(infile)
            self.inputSize = result[0]
            weights_list_cpu = result[1]
            self.weights_list = []
            for w in weights_list_cpu:
                self.weights_list.append(cp.asarray(w))
            bias_list_cpu = result[2]
            self.bias_list = []
            for b in bias_list_cpu:
                self.bias_list.append(cp.asarray(b))

    def set_gammas(self, g_w=None, g_b=None):
        if g_w != None:
            self.gamma_w = g_w
        if g_b != None:
            self.gamma_b = g_b

    def feed_forward(self, input: cp.array) -> cp.array:
        if input.shape[0] != self.inputSize:
            raise ValueError("Input Size error!")

        self.nodes_list = []
        self.last_input = input[cp.newaxis].T
        # self.last_input = cp.asarray(input[np.newaxis].T)

        middleArr = None
        for i, w in enumerate(self.weights_list):
            if i == 0:
                middleArr = w @ self.last_input
            else:
                middleArr = w @ middleArr
            middleArr += self.bias_list[i]
            if i == len(self.weights_list)-1:
                middleArr = self.activation_func_outside(middleArr)
            else:
                middleArr = self.activation_func_inside(
                    middleArr, self.__random_matrix(1.0, middleArr.shape))
            self.nodes_list.append(middleArr)
        return self.nodes_list[-1]

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
            self.weights_list[i] = w - self.gamma_w * derivative

        for i in range(len(self.bias_list)):
            self.bias_list[i] -= self.gamma_b * deltas[i + 1]


if __name__ == "__main__":
    nn = StochNN_GPU(2, [3200], 1)
    xor_map = [(cp.array([1.0, 1.0], dtype=cp.single),
                cp.array([0.0], dtype=cp.single)),
               (cp.array([1.0, 0.0], dtype=cp.single),
                cp.array([1.0], dtype=cp.single)),
               (cp.array([0.0, 1.0], dtype=cp.single),
                cp.array([1.0], dtype=cp.single)),
               (cp.array([0.0, 0.0], dtype=cp.single),
                cp.array([0.0], dtype=cp.single)),
               ]

    import time
    begin = time.time()
    for _ in range(10000):
        for input, output in xor_map:
            nn.feed_forward(input)
    end = time.time()
    print(end - begin)

    # # learning
    # learning_curve = []
    # MEASURE_POINTS = 10
    # # EPOCHS = 6000
    # EPOCHS = 800
    # SINGLE_AVERAGE = 20
    # for _ in range(EPOCHS):
    #     average = []
    #     for _ in range(MEASURE_POINTS):
    #         for input, output in xor_map:
    #             predictions = []
    #             for _ in range(SINGLE_AVERAGE):
    #                 predictions.append(nn.feed_forward(input))
    #             pred_avg = np.array(predictions).mean()
    #             nn.backpropagation(output, pred_avg)
    #             average.append((pred_avg - output)**2)
    #     learning_curve.append(np.array(average).mean())
    #     if learning_curve[-1] < 0.15:
    #         nn.set_gammas(0.005, 0.005)
    #     if learning_curve[-1] < 0.02:
    #         break

    # for input, output in xor_map:
    #     ps = []
    #     for i in range(10):
    #         ps.append(nn.feed_forward(input))
    #     print(input, output, np.array(ps).mean())

    # nn.save_model("test.stoch")

    # from matplotlib import pyplot as plt
    # plt.plot(learning_curve)
    # plt.show()
