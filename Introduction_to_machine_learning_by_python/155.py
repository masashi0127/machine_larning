import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, linear_model, cluster
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


data = np.c_[data_x, data_vy]
model = cluster.KMeans(n_clusters=3)
model.fit(data)

labels = model.labels_

plt.scatter(data_x[labels == 0], data_vy[labels == 0], c='black', s=30, marker='^', label='cluster 0')
plt.scatter(data_x[labels == 1], data_vy[labels == 1], c='black', s=30, marker='x', label='cluster 1')
plt.scatter(data_x[labels == 2], data_vy[labels == 2], c='black', s=30, marker='*', label='cluster 2')

plt.plot(data_x, data_ty, linestyle=':', label='non noise curve')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.show()
