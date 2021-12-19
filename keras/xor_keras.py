from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score

X_train = np.array([
  [1.0, 1.0],
  [1.0, 0.0],
  [0.0, 1.0],
  [0.0, 0.0]
])

y_train = np.array([
  [0.0],
  [1.0],
  [1.0],
  [0.0]
])

model = Sequential()
model.add(Dense(3, input_dim=2, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="mse",
              optimizer='adam', metrics=['mse'])

history = model.fit(X_train, y_train, epochs=2000, batch_size=1, verbose=1)

testResults = model.predict(X_train)


print(X_train)
print(testResults)
# print(confusion_matrix(y_train.argmax(axis=1), testResults.argmax(axis=1)))
# print(classification_report(y_train.argmax(axis=1), testResults.argmax(axis=1)))
# print("Cohen's Kappa: {}".format(cohen_kappa_score(
#     y_train.argmax(axis=1), testResults.argmax(axis=1))))
# print("Accuracy: ", accuracy_score(
#     y_train.argmax(axis=1), testResults.argmax(axis=1)))
