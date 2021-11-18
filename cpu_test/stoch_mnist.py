from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from StochNN import StochNN
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

model = StochNN(num_pixels, [3*num_pixels, 2*num_pixels], num_classes)

for i in range(4):
    print(f"epoch: {i}")
    for i, t in enumerate(X_train):
        predictions = []
        res = model.feed_forward(t)
        model.backpropagation(y_train[i], res)
        print(i)

model.save_model("mnist.stoch")
