import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from StochNN_GPU import StochNN_GPU
import time
import cupy as cp
import numpy as np

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

model = StochNN_GPU(num_pixels, [2*num_pixels, 2*num_pixels], num_classes)
model.load_model("test_mnists/mnist_81proc.stoch")

accuracy = cp.zeros(1, dtype=cp.int32)
y_result = []

begin = time.time()
for i, x in enumerate(X_test_gpu):
    pred = model.feed_forward(x)
    y_result.append(pred.get())
    guess = cp.argmax(pred)
    target = cp.argmax(y_test_gpu[i])
    if target == guess:
        accuracy += 1
print(f"ff time: {time.time() - begin}")

y_result = np.array(y_result)

print("kappa:", cohen_kappa_score(y_test.argmax(axis=1), y_result.argmax(axis=1)))
print("acc:", accuracy.get() / y_test.shape[0])
disp = ConfusionMatrixDisplay(
    confusion_matrix(y_test.argmax(axis=1), y_result.argmax(axis=1)))
disp.plot(colorbar=False)
plt.show()
