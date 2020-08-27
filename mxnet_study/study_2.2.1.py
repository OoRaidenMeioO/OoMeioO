"""
动手学深度学习
第二章代码
2.2.1-2.2.2
高伟 2020.8.26
"""
# 创建数组 ndarray，shape形状，size元素总数
from mxnet import nd
x = nd.arange(12)
print(x)
print(x.shape)
print(x.size)

# 更改形状 reshape
X = x.reshape((3, 4))
print(X)
print(X.shape)
print(X.size)
Y = X.reshape((-1, 3))
print(Y)

# 特殊张量 zeros 全零矩阵 ones 全一矩阵
print(nd.zeros((2, 3, 4)))
print(nd.ones((3, 4)))

# 创建array list转
X = nd.array([[2, 3, 4], [3, 5, 6]])
print(X)

# 随机生成arrray 随机生成正态分布
Y = nd.random.normal(0, 1, shape=(3, 4))
print(Y)

# 基本运算,按元素运算operator
X = nd.ones((3, 4))
print(X+Y)
print(X*Y)
print(X/Y)

# 按元素进行指数运算
print(Y.exp())

# 矩阵乘法
print(nd.dot(X, Y.T))

# 矩阵连接 维度0，列连接，维度1，行连接
print(nd.concat(X, Y, dim=0))
print(nd.concat(X, Y, dim=1))

# 判断两矩阵元素是否相同
print(X == Y)

# 矩阵所有元素求和
print(X.sum())

# asscalar将结果变为标量
print(X.norm().asscalar())