from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from Stoch import Stochastic

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score

(X_train, y_train), (X_test, y_test) = mnist.load_data()


num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()
model.add(Stochastic(2*num_pixels, input_dim=num_pixels))
model.add(Stochastic(2*num_pixels))
# model.add(Dense(num_pixels, activation='sigmoid'))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='SGD', metrics=['accuracy'])
# model.optimizer.lr.assign(0.0005)

history = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=64, batch_size=60, verbose=1)

testResults = model.predict(X_test)

print(confusion_matrix(y_test.argmax(axis=1), testResults.argmax(axis=1)))
print(classification_report(y_test.argmax(axis=1), testResults.argmax(axis=1)))
print("Cohen's Kappa: {}".format(cohen_kappa_score(
    y_test.argmax(axis=1), testResults.argmax(axis=1))))
print("Accuracy: ", accuracy_score(
    y_test.argmax(axis=1), testResults.argmax(axis=1)))

model.save("keras_model.h5")
