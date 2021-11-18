from StochNN import StochNN
import numpy as np

nn = StochNN(2, [32], 1)

nn.load_model("test.stoch")

xor_map = [(np.array([1.0, 1.0]), np.array([0.0])),
           (np.array([1.0, 0.0]), np.array([1.0])),
           (np.array([0.0, 1.0]), np.array([1.0])),
           (np.array([0.0, 0.0]), np.array([0.0])),
           ]

for input, output in xor_map:
    ps = []
    for i in range(10):
        ps.append(nn.feed_forward(input))
    print(input, output, np.array(ps).mean())
