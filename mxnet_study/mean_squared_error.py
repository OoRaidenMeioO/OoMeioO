"""
均方误差损失函数的实现
mean squared error
高伟 2020.9.4
"""

import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


y_one_hot = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y1), np.array(y_one_hot)))

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y2), np.array(y_one_hot)))
