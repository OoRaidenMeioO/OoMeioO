"""
使用numpy实现简单神经网络
数据集mnist
高伟 2020.9.4
"""
import numpy as np

# 把labal=[n],转换为onehot表示
# enumerate(),列出索引序列（列表，字符串，元组）的数据和下标
def change_one_hot_label(x):
    T = np.zeros((x.size, 10))
    for index, row in enumerate(T):
        row[x[index]] = 1
    return T

# 设计权重矩阵
# network["W1"]: 100*50  network["W2"]: 50*10
# network["b1"]: 50*1    network["W3"]: 10*1

W1 = np.random.randint(0, 10, size=(100, 50))
W2 = np.random.randint(0, 10, size=(50, 10))
b1 = np.random.randint(0, 10, size=(50,))
b2 = np.random.randint(0, 10, size=(10,))
# 设计数据
# x_train:(1000,50) y_train:(1000,10)  1000数据
# x_test: (100, 50) y_test: (100,10)   100数据
x_train = np.random.randint(0, 10, size=(1000, 100))
y_train = np.dot(np.dot(x_train, W1)+b1, W2)+b2
x_test = np.random.randint(0, 10, size=(100, 100))
y_test = np.dot(np.dot(x_test, W1)+b1, W2)+b2

print(y_test)
print(y_test.shape)



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

print(change_one_hot_label(np.array([6])))

# 批处理版本
# np.tile 复制
def softmax_batch(x):
    max_x = np.max(x, axis=1)
    exp_x = np.exp(x-np.tile(max_x, (x.shape[1], 1)).T)
    sum_x = np.sum(exp_x, axis=1)
    return exp_x/(np.tile(sum_x, (x.shape[0], 1))).T

print(x_test.shape)
print(x_test.shape[1])
print(np.tile(x_test, (x_test.shape[1], 1)).T.shape)
# print(np.tile(x_test, (x_test.shape[1], 1)).T))

y = softmax_batch(x_test)
# print("y:", y)