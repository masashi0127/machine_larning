import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# x = np.random.rand(100, 1)
# x = x * 4 - 2
#
# y = 3 * x - 2
#
# y += np.random.randn(100, 1)
#
# model = linear_model.LinearRegression()
# model.fit(x, y)
#
# print(model.coef_)
# print(model.intercept_)
# print(model.score(x, y))
#
# plt.scatter(x, y, marker='+')
# plt.scatter(x, model.predict(x), marker='o')
# plt.show()

# x = np.random.rand(100, 1)
#
# x = x * 4 - 2
#
# y = 3 * x ** 2 - 2
#
# y += np.random.randn(100, 1)
#
# model = linear_model.LinearRegression()
# model.fit(x ** 2, y)
#
# print(model.coef_)
# print(model.intercept_)
# print(model.score(x ** 2, y))
#
# plt.scatter(x, y, marker='+')
# plt.scatter(x, model.predict(x ** 2), marker='o')
# plt.show()

x1 = np.random.randn(100, 1)
x1 = x1 * 4 - 2
x2 = np.random.randn(100, 1)
x2 = x2 * 4 - 2
y = 3 * x1 - 2 * x2 + 1
y += np.random.randn(100, 1)

x1_x2 = np.c_[x1, x2]

model = linear_model.LinearRegression()
model.fit(x1_x2, y)

y_ = model.predict(x1_x2)

plt.subplot(1, 2, 1)
plt.scatter(x1, y, marker='+')
plt.scatter(x1, y_, marker='o')
plt.xlabel('x1')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.scatter(x2, y, marker='+')
plt.scatter(x2, y_, marker='o')
plt.xlabel('x2')
plt.ylabel('y')

print(model.coef_)
print(model.intercept_)
print(model.score(x1_x2, y))

plt.tight_layout()
plt.show()
