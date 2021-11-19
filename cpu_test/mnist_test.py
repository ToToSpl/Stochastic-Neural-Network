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
model.load_model("mnist_5.stoch")

accuracy = 0.0
acc_matrix = np.zeros((num_classes, num_classes))

for i, x in enumerate(X_test):
    guess = np.argmax(model.feed_forward(x))
    target = np.argmax(y_test[i])
    acc_matrix[target, guess] += 1
    if target == guess:
        accuracy += 1

print(acc_matrix)
print(accuracy / y_test.shape[0])
