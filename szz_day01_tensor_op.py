# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/13 
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day01_tensor_op.py
"""
import tensorflow as tf

# tf.constant，tf.add 都是操作
# op是一个载体，tensor是具体数值
con_a = tf.constant(3.0, name='con_a')  # 把3.0装进去
con_b = tf.constant(4.0, name='con_b')  # 把4.0装进去

sum_c = tf.add(con_a, con_b, name='sum_c')

# 打印出来的结果都是tensor，类似为ndarray
print("打印con_a：\n", con_a)  # Tensor("Const:0", shape=(), dtype=float32)
print("打印con_b：\n", con_b)
print("打印sum_c：\n", sum_c)

# config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# 用于查看操作所在的设备运行信息
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
#     file_writer = tf.summary.FileWriter('./tmp/summary', graph=sess.graph)
#     c_res = sess.run(sum_c)
#     print(c_res)

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./tmp/summary', graph=sess.graph)
    a_res, b_res, c_res = sess.run([con_a, con_b, sum_c])
    print(a_res, b_res, c_res)