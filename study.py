"""
机器学习线性代数基础：Python语言描述
第一章代码
高伟 2020.8.25
"""
import numpy as np

# S_1  行向量描述 a = [1 2 3 4]
s1 = np.array([1, 2, 3, 4])
print("S_1:")
print(s1)

# S_2 行向量转列向量 转置
# 需用（1，x）的矩阵才可以转置
s2 = np.array([[1, 2, 3, 4]])
print("S_2:")
print(s2)
print(s2.T)

# S_3 向量加法
s3_u = np.array([[1, 2, 3]]).T
s3_v = np.array([[5, 6, 7]]).T
print("S_3:")
print(s3_u + s3_v)

# S_4 向量的数乘乘法
# 几何意义：延向量方向拉伸相应倍数
s4 = np.array([[1, 2, 3]]).T
print("S_4:")
print(3*s4)

# S_5 向量间的乘法：内积
# 对应位置元素相乘，相加
# u*v = |u||v|cosθ