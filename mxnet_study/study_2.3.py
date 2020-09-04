"""
动手学深度学习
第二章代码
2.3-
高伟 2020.8.26
"""
from mxnet import autograd, nd
import numpy as np
# #
# x = nd.arange(4).reshape((4, 1))
# x2 = nd.arange(4).reshape((1, 4))
# # 申请梯度所需内存
# x.attach_grad()
#
# with autograd.record():
#     y = 2*nd.dot(x.T, x)+5*nd.dot(x.T, x)
# y.backward()
# # norm() L2范数
# # x.grad就是x的梯度值
# assert(x.grad - 4*x-10*x).norm().asscalar() == 0
# print(x.grad)

# BP算法简单实现
x = nd.arange(8).reshape((4, 2))
b = nd.array([[1, 1], [1, 1]])
k = 2
y = k*nd.dot(x.T, x) + b

# # 损失函数简单实现
# def loss_01():
#
# # 常用损失函数：0-1损失函数、平方损失函数、绝对损失函数、对数损失函数、交叉熵损失函数
# # 对数损失函数
# def loss_log(y_true, y_pred):
#     y_true = nd.array(y_true)
#     y_pred= nd.array(y_pred)
#     # 判断两者形状是否相同
#     assert(len(y_true) == len(y_pred))

# 无批处理
# sigmoid函数,激活函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# softmax函数,归一化函数
# 重点：消除上溢和下溢，减去最大值进行运算
def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x-max_x)
    softmax_x = exp_x/np.sum(exp_x)
    return softmax_x

def predict






