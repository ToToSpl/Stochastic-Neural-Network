from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from StochNN import StochNN
import numpy as np

EPOCHS = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

model = StochNN(num_pixels, [3*num_pixels], num_classes)

for i in range(EPOCHS):
    print("new epoch")
    for j, t in enumerate(X_train):
        predictions = []
        res = model.feed_forward(t)
        model.backpropagation(y_train[j], res)
        if j % 100 == 0:
            print(f"epoch: {i}\t{j/X_train.shape[0]}%...")
    model.save_model(f"mnist_{i}")

model.save_model("mnist_final.stoch")
