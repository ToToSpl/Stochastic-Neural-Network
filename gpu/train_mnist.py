from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from StochNN_GPU import StochNN_GPU
from sklearn.utils import shuffle
import cupy as cp
import time

EPOCHS = 16
LOG_RATE = 500

(X_train, y_train), _ = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')

X_train = X_train / 255

y_train = to_categorical(y_train)

y_train, X_train = shuffle(y_train, X_train)
train_size = X_train.shape[0]

X_train_gpu = []
y_train_gpu = []

begin = time.time()
for i, x in enumerate(X_train):
    X_train_gpu.append(cp.asarray(x, dtype=cp.float32))
    y_train_gpu.append(cp.asarray(y_train[i], dtype=cp.float32))
print(f"load time: {time.time() - begin}")


num_classes = y_train.shape[1]

model = StochNN_GPU(num_pixels, [int(2*num_pixels), int(2*num_pixels)], num_classes)
model.load_model("mnist_point.stoch")
model.set_gammas(0.000005, 0.000005)


for i in range(EPOCHS):
    print("new epoch")
    begin = time.time()
    acc = cp.zeros(1, dtype=cp.int32)
    for j, t in enumerate(X_train_gpu):
        predictions = []
        res = model.feed_forward(t)
        if cp.argmax(res) == cp.argmax(y_train_gpu[j]):
            acc += 1
        model.backpropagation(y_train_gpu[j], res)
        if j % LOG_RATE == 0 and j != 0:
            proc = j/train_size * 100
            accuracy = acc.get() / LOG_RATE
            print("epoch: {}\tacc: {:.2f}\tproc: {:.1f}".format(
                i, accuracy[0], proc))
            acc = cp.zeros(1, dtype=cp.int32)
    print(f"epoch time: {time.time() - begin}")
    model.save_model(f"mnist_{i}.stoch")

