import numpy as np
import matplotlib.pyplot as plt

AVG_VALUE = 20

def random_tanh(x):
    val = 1 / (1 + np.exp(-x))
    rand = np.random.rand(*x.shape)
    return 2.0 * (val > rand) - 1.0


x_vec = np.arange(-5, 5, step=0.5)
y_vec = []
for x in x_vec:
    y_tot = 0.0
    i = x * np.ones(AVG_VALUE)
    y_tot += random_tanh(i)
    y_vec.append(y_tot.mean())
y_vec = np.array(y_vec)


plt.title(f"Probing: {AVG_VALUE}")
plt.plot(x_vec, np.tanh(0.5 * x_vec), 'r--', label="tanh")
plt.scatter(x_vec, y_vec, label="rand activation")
plt.legend(loc="upper left")
plt.show()