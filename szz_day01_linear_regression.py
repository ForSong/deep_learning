# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/19 
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day01_linear_regression.py
"""
import numpy as np


def basic_sigmoid(x):
    """
    计算sigmoid函数的值
    :return: 计算结果
    """
    s = 1 / (1 - np.exp(-x))
    return s


def propagate(w, b, x, y):
    """
    这个方法的作用是用于求解梯度和损失
    :param w: 权重
    :param b: 偏置
    :param x: 数据
    :param y: 数据
    :return: 损失cost,参数w的梯度dw，参数b的梯度db
    """
    m = x.shape[1]
    # 从前往后计算损失（前向传播）
    a = basic_sigmoid(np.dot(w.T, x) + b)
    # 计算损失
    cost = -1 / m * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    # 从后往前求出梯度（反向传播）

def model(x_train,x_test,y_train,y_test,num_iteration=2000,learning_rate=0.0001):
    """
    主要函数
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param num_iteration:
    :param learning_rate:
    :return:
    """
    # 修改数据形状
    x_train = x_train.reshape(-1,x_train.shape[0])
    x_test = x_test.reshape(-1,x_test.shape[0])
    y_train = y_train.reshape()
