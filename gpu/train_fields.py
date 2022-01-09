import numpy as np
import cupy as cp
import pickle5 as pickle
import matplotlib.pyplot as plt
from StochNN_GPU import StochNN_GPU
from sklearn.utils import shuffle

X_train = None
y_train = None
with open("follower_circle.pckl", "rb") as infile:
    result = pickle.load(infile)
    X_train = result[0]
    y_train = result[1]

X_train, y_train = shuffle(X_train, y_train)

y_train = (y_train + np.ones(y_train.shape)) * 0.5

X_train_gpu = []
y_train_gpu = []
for i in range(X_train.shape[0]):
    X_train_gpu.append(cp.array(X_train[i], dtype=cp.float32))
    y_train_gpu.append(cp.array(y_train[i], dtype=cp.float32))

model = StochNN_GPU(2, [16], 2)
model.set_gammas(0.005, 0.005)

for i in range(64):
    for j, t in enumerate(X_train_gpu):
        pred = model.feed_forward(t)
        model.backpropagation(y_train_gpu[j], pred)
    print(i)

y_pred = []
for t in X_train_gpu:
    temp = model.feed_forward(t)
    for _ in range(99):
        temp += model.feed_forward(t)
    y_pred.append(temp * 0.01)

y_pred = cp.array(y_pred)
y_pred = 2.0 * y_pred - cp.ones(y_pred.shape)
y_train = 2.0 * y_train - np.ones(y_train.shape)
y_pred = y_pred.get()

plt.quiver(X_train[:, 0], X_train[:, 1], y_train[:, 0], y_train[:, 1], color='b', label='vector field')
plt.quiver(X_train[:, 0], X_train[:, 1], y_pred[:, 0], y_pred[:, 1], color='r', label='average network output')
plt.legend()
plt.show()


# STEP_SIZE = 0.1
# STEPS = 5000

# start_point = cp.array([0.5, 0.5], dtype=cp.float32)

# positions = cp.zeros((STEPS, 2), dtype=cp.float32)
# positions[0] = start_point
# rand_positions = np.zeros((STEPS, 2))
# rand_start_point = np.array([0.5, 0.5])
# rand_positions[0] = start_point.get()
# for i in range(1, STEPS):
#     vec = model.feed_forward(positions[i - 1])
#     vec = 2.0 * vec - 1.0
#     next_pos = positions[i - 1] + STEP_SIZE * vec.reshape((1,2))
#     positions[i] = next_pos.reshape((2))
#     rand_vec = 2.0 * np.random.rand(2) - 1.0
#     rand_vec = rand_vec / np.linalg.norm(rand_vec)
#     rand_vec *= cp.linalg.norm(vec).get()
#     next_pos = rand_positions[i - 1] + STEP_SIZE * rand_vec.reshape((1,2))
#     rand_positions[i] = next_pos.reshape((2))

# positions = positions.get()

    

# plt.plot(rand_positions[:, 0], rand_positions[:, 1], color='r', label="random direction")
# plt.plot(positions[:, 0], positions[:, 1], color='b', label="NN set direction")
# plt.legend(loc="upper right")
# plt.show()
