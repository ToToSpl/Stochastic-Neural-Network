import random
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

_, (X_test, y_test) = mnist.load_data()

indexes_r = [random.randint(0, y_test.shape[0] - 1) for _ in range(4)]


f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(X_test[indexes_r[0]])
axarr[0,0].set_title(y_test[indexes_r[0]])
axarr[0,0].axis('off')

axarr[0,1].imshow(X_test[indexes_r[1]])
axarr[0,1].set_title(y_test[indexes_r[1]])
axarr[0,1].axis('off')

axarr[1,0].imshow(X_test[indexes_r[2]])
axarr[1,0].set_title(y_test[indexes_r[2]])
axarr[1,0].axis('off')

axarr[1,1].imshow(X_test[indexes_r[3]])
axarr[1,1].set_title(y_test[indexes_r[3]])
axarr[1,1].axis('off')

plt.show()