import argparse
import math
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True, help="file path of loged data")
args = vars(ap.parse_args())

acc = []
epochs = []
acc_l = []
epochs_l = []

with open(args['file']) as f:
    for l in f:
        splitted = l.split("\t")
        if len(splitted) == 1:
            continue
        e = float(splitted[0].split()[1])
        a = float(splitted[1].split()[1])
        p = float(splitted[2].split()[1])
        x_point = e + p / 100.0
        acc.append(a)
        epochs.append(x_point)

acc_l.append(acc[0])
epochs_l.append(math.floor(epochs[0]))
for i in range(1, len(epochs)):
    if math.floor(epochs[i]) == epochs_l[-1]:
        continue
    epochs_l.append(math.floor(epochs[i]))
    acc_l.append(acc[i - 1])
acc_l.append(acc[-1])
epochs_l.append(math.ceil(epochs[-1]))

titleLR = 1e-4
plt.title(f"Training at learning rate: {titleLR}")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(epochs, acc, label="accuracy during epoch")
plt.plot(epochs_l, acc_l, label="epoch final accuracy")
plt.legend(loc="upper left")
plt.show()
