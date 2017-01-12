import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, linear_model
from sklearn.metrics import confusion_matrix, accuracy_score

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

CLASS_RADIUS = 0.6
labels = (data_x ** 2 + data_vy ** 2) < CLASS_RADIUS ** 2

label_train = labels[idx_train]
label_test = labels[idx_test]

data_train = np.c_[x_train, y_train]
data_test = np.c_[x_test, y_test]

X1_TRAIN = x_train
X1_TEST = x_test

model = linear_model.LinearRegression()
model.fit(X1_TRAIN, y_train)

plt.plot(x_test, model.predict(X1_TEST), linestyle='-.', label='ploy deg 1')

X2_TRAIN = np.c_[x_train ** 2, x_train]
X2_TEST = np.c_[x_test ** 2, x_test]

model = linear_model.LinearRegression()
model.fit(X2_TRAIN, y_train)

plt.plot(x_test, model.predict(X2_TEST), linestyle='--', label='ploy deg 2')

X9_TRAIN = np.c_[x_train ** 9, x_train ** 8, x_train ** 7, x_train ** 6, x_train ** 5, x_train ** 4, x_train ** 3, x_train ** 2, x_train]
X9_TEST = np.c_[x_test ** 9, x_test ** 8, x_test ** 7, x_test ** 6, x_test ** 5, x_test ** 4, x_test ** 3, x_test ** 2, x_test]

model = linear_model.LinearRegression()
model.fit(X9_TRAIN, y_train)

plt.plot(x_test, model.predict(X9_TEST), linestyle='--', label='ploy deg 9')

plt.scatter(x_train, y_train, c='black', s=30, marker='v', label='train')
plt.scatter(x_test, y_test, c='black', s=30, marker='x', label='test')

plt.plot(data_x, data_ty, linestyle=':', label='non noise curve')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.show()
