# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/14 
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day01_tensor_shape.py
"""
import tensorflow as tf

# 形状变化

a_p = tf.placeholder(dtype=tf.float32, shape=[None, None])
b_p = tf.placeholder(tf.float32, shape=[None, 10])
c_p = tf.placeholder(tf.float32, shape=[2, 3])

print('a_p shape', a_p.get_shape())
print('b_p shape', b_p.get_shape())
print('c_p shape', c_p.get_shape())
# 输出
# a_p shape (?, ?)
# b_p shape (?, 10)
# c_p shape (2, 3)


# 静态形状
a_p.set_shape([5, 6])
print('a_p shape', a_p.get_shape())
# 1. 对于张量形状已经固定的，不能再次修改
# 2. 张量形状不能跨维度修改
# a_p.set_shape([3, 4]) 报错
# a_p.set_shape([30]) 报错
# print('a_p shape', a_p.get_shape())

# 动态形状，相当于创造一个新的张量
# 注意改变形状的时候数量要保持和原来相同
# reshape会创建一个新的张量，注意这点
c_p_reshape = tf.reshape(c_p, [3, 2])
print('c_p_reshape', c_p_reshape)

