# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/14 
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day01_tensor.py
"""
import tensorflow as tf

con_1 = tf.constant(3.0)  # ()
con_2 = tf.constant([1, 2, 3, 4])  # (4,)
con_3 = tf.constant([[1, 2], [3, 4]])  # (2, 2)
con_4 = tf.constant([[[1, 2], [3, 4]], [[4, 2], [4, 6]]])  # (2, 2, 2)



print(con_1.shape, con_2.shape, con_3.shape, con_4.shape)
