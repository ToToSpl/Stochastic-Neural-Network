from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

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
model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
model.add(Dense(0.5*num_pixels, activation='sigmoid'))
model.add(Dense(0.5*num_pixels, activation='sigmoid'))
model.add(Dense(0.5*num_pixels, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=64, batch_size=200, verbose=1)

testResults = model.predict(X_test)

print(confusion_matrix(y_test.argmax(axis=1), testResults.argmax(axis=1)))
print(classification_report(y_test.argmax(axis=1), testResults.argmax(axis=1)))
print("Cohen's Kappa: {}".format(cohen_kappa_score(
    y_test.argmax(axis=1), testResults.argmax(axis=1))))
print("Accuracy: ", accuracy_score(
    y_test.argmax(axis=1), testResults.argmax(axis=1)))

'''
[[ 968    1    0    2    2    0    2    1    3    1]
 [   0 1128    2    1    1    0    1    0    2    0]
 [   3    4 1007    7    4    0    2    4    1    0]
 [   0    1    1 1000    0    2    0    2    1    3]
 [   1    0    1    1  972    0    1    0    0    6]
 [   1    0    0   15    2  864    5    0    5    0]
 [   3    3    2    1   14    2  931    0    2    0]
 [   1    5    8    4    4    0    0  996    4    6]
 [   3    1    3    4    4    2    4    3  945    5]
 [   2    3    0    4   13    1    0    2    1  983]]
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       980
           1       0.98      0.99      0.99      1135
           2       0.98      0.98      0.98      1032
           3       0.96      0.99      0.98      1010
           4       0.96      0.99      0.97       982
           5       0.99      0.97      0.98       892
           6       0.98      0.97      0.98       958
           7       0.99      0.97      0.98      1028
           8       0.98      0.97      0.98       974
           9       0.98      0.97      0.98      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000

Cohen's Kappa: 0.9771013573070414
Accuracy:  0.9794[[ 971    1    0    0    0    1    3    0    1    3]
 [   1 1128    1    0    0    0    2    1    2    0]
 [   1    1 1016    3    1    0    1    3    6    0]
 [   0    0    4  993    0    4    0    3    2    4]
 [   1    0    1    0  967    0    4    2    1    6]
 [   2    0    0    6    1  878    2    0    2    1]
 [   2    3    1    1    1    3  947    0    0    0]
 [   2    1   11    0    0    0    0 1008    1    5]
 [   0    0    4    2    1    3    2    2  957    3]
 [   2    2    0    1    7    3    0    1    1  992]]
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.98      0.98      1032
           3       0.99      0.98      0.99      1010
           4       0.99      0.98      0.99       982
           5       0.98      0.98      0.98       892
           6       0.99      0.99      0.99       958
           7       0.99      0.98      0.98      1028
           8       0.98      0.98      0.98       974
           9       0.98      0.98      0.98      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000

Cohen's Kappa: 0.9841049152070588
Accuracy:  0.9857
'''