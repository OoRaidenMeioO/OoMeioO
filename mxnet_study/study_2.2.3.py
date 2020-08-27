"""
动手学深度学习
第二章代码
2.2.3-2.2.6
高伟 2020.8.26
"""
from mxnet import nd
import numpy as np

# 广播机制,元素复制
A = nd.arange(3).reshape(3, 1)
B = nd.arange(2).reshape(1, 2)
print(A)
print(B)
print(A+B)

# 索引 左闭右开
X = nd.random.normal(0, 1, shape=(3, 4))
print(X)
print(X[1:2])
print(X[1, 2])
print(X[1, 2:3])

# 内存开销

# numpy和ndarray相互变换
P = np.ones((2,3))
D = nd.array(P)
print(D)
P = D.asnumpy()
print(P)


