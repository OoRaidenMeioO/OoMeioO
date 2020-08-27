"""
动手学深度学习
第二章代码
2.3-
高伟 2020.8.26
"""
from mxnet import autograd, nd
#
x = nd.arange(4).reshape((4, 1))
# 申请梯度所需内存
x.attach_grad()