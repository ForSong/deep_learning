# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/14 
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day01_variable.py
变量相关的op
"""
import tensorflow as tf

# 这种变量op需要进行手动初始化
a = tf.Variable(initial_value=30.0)
b = tf.Variable(initial_value=40.0)

sum_ab = tf.add(a, b)

# 初始化op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 手动运行init_op
    sess.run(init_op)
    print(sess.run(sum_ab))
