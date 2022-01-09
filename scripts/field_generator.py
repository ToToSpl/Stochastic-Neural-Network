import pickle
import numpy as np
import matplotlib.pyplot as plt

FIELD_CENTER = np.array([0.5, 0.5])
STEP_SIZE = 0.05


def field_center(i, j):
    vec_dir = np.array([-j + FIELD_CENTER[0], -i +
                       FIELD_CENTER[1]], dtype=np.float32)
    if np.linalg.norm(vec_dir) == 0:
        vec_dir = np.array([0.0, 0.0])
    else:
        vec_dir = vec_dir / np.linalg.norm(vec_dir)
    return vec_dir


def field_circle(i, j):
    vec_dir = np.array(
        [-i + FIELD_CENTER[0], j - FIELD_CENTER[1]], dtype=np.float32)
    if np.linalg.norm(vec_dir) == 0:
        vec_dir = np.array([0.0, 0.0])
    else:
        vec_dir = vec_dir / np.linalg.norm(vec_dir)
    return vec_dir


x_vecs = []
y_vecs = []
for i in np.arange(0, 1 + STEP_SIZE, STEP_SIZE):
    for j in np.arange(0, 1 + STEP_SIZE, STEP_SIZE):
        vec = np.array([j, i], dtype=np.float32)
        x_vecs.append(vec)
        vec_dir = field_circle(i, j)
        y_vecs.append(vec_dir)

x_vecs = np.array(x_vecs)
y_vecs = np.array(y_vecs)

with open("follower_circle.pckl", 'wb') as outfile:
    pickle.dump([x_vecs, y_vecs], outfile, pickle.HIGHEST_PROTOCOL)

plt.quiver(x_vecs[:, 0], x_vecs[:, 1], y_vecs[:, 0], y_vecs[:, 1])
plt.show()
