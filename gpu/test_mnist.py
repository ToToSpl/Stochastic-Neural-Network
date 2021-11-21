from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from StochNN_GPU import StochNN_GPU
import time
import cupy as cp

_, (X_test, y_test) = mnist.load_data()
num_pixels = X_test.shape[1] * X_test.shape[2]

X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
X_test = X_test / 255

y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

X_test_gpu = []
y_test_gpu = []

begin = time.time()
for i, x in enumerate(X_test):
    X_test_gpu.append(cp.asarray(x, dtype=cp.float32))
    y_test_gpu.append(cp.asarray(y_test[i], dtype=cp.float32))
print(f"load time: {time.time() - begin}")

model = StochNN_GPU(num_pixels, [3*num_pixels, 2*num_pixels], num_classes)
model.load_model("test_mnists/mnist_66.stoch")

accuracy = cp.zeros(1, dtype=cp.int32)
# acc_matrix = np.zeros((num_classes, num_classes))

begin = time.time()
for i, x in enumerate(X_test_gpu):
    pred = model.feed_forward(x)
    guess = cp.argmax(pred)
    target = cp.argmax(y_test_gpu[i])
    # acc_matrix[target, guess] += 1
    if target == guess:
        accuracy += 1
print(f"ff time: {time.time() - begin}")

# print(acc_matrix)
print(accuracy.get() / y_test.shape[0])
