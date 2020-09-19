# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/14 
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day01_placeholder.py
"""
import tensorflow as tf

# 如果值不确定的时候可以定义占位符
con_a = tf.constant(4.0)
a = tf.placeholder(tf.float32, name='a')
b = tf.placeholder(tf.float32, name='b')

sum_ab = tf.add(a, b)

# 定义两个placeholder
plt_a = tf.placeholder(dtype=tf.float32)
plt_b = tf.placeholder(dtype=tf.float32)

plt_add = tf.add(plt_a, plt_b)

# 开启会话
with tf.Session() as sess:
    # print('占位符的结果：\n', sess.run(sum_ab, feed_dict={a: 3.0, b: 4.0}))
    # 在运行的时候填充数据，使用feed_dict{:,:}填充
    # 一般用在数据不确定的时候
    res = sess.run(plt_add, feed_dict={plt_a: 5.0, plt_b: 6.0})
    # eval也需要在由session的上下文环境中运行，也可以直接打印出值
    print(con_a.eval())
    print(res)
