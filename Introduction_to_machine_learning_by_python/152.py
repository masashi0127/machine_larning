import matplotlib.pyplot as plt
import numpy as np

x_max = 1
x_min = -1

y_max = 2
y_min = -1

SCALE = 50
TEST_RATE = 0.3

data_x = np.arange(x_min, x_max, 1 / float(SCALE)).reshape(-1, 1)

data_ty = data_x ** 2
data_vy = data_ty + np.random.randn(len(data_ty), 1) * 0.5

def split_train_test(array):
    length = len(array)
    n_train = int(length * (1 - TEST_RATE))

    indices = list(range(length))
    np.random.shuffle(indices)
    idx_train = indices[:n_train]
    idx_test = indices[n_train:]

    return sorted(array[idx_train]), sorted(array[idx_test])

indices = np.arange(len(data_x))
idx_train, idx_test = split_train_test(indices)

x_train = data_x[idx_train]
y_train = data_vy[idx_train]

x_test = data_x[idx_test]
y_test = data_vy[idx_test]

plt.scatter(data_x, data_vy, label='target')

plt.plot(data_x, data_ty, linestyle=':', label='non noise curve')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.show()
