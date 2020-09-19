# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/13
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day01.py
"""
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error

# 实现一个加法运算
new_a = tf.constant(11.0)
b = tf.constant(20.0)

c = tf.add(new_a, b)

# 获取默认图
g = tf.get_default_graph()
# 结果是一个object对象，一段内存空间，被TensorFlow定义为了一个对象
# 存储了之前的操作和张量的图
# print('获取当前加法运算的图', g) # <tensorflow.python.framework.ops.Graph object at 0x118424b10>

# 打印所有操作和张量的默认图
# 可以看到和g一样
# print(a.graph) # <tensorflow.python.framework.ops.Graph object at 0x118424b10>
# print(b.graph) # <tensorflow.python.framework.ops.Graph object at 0x118424b10>
# print(c.graph) # <tensorflow.python.framework.ops.Graph object at 0x118424b10>

# 创建另外一张图
new_g = tf.Graph()
with new_g.as_default():
    new_a = tf.constant(11.0)
    new_b = tf.constant(20.0)

    new_c = tf.add(new_a, new_b)

print(new_a.graph)  # <tensorflow.python.framework.ops.Graph object at 0x12bed5dd0>
print(new_b.graph)  # <tensorflow.python.framework.ops.Graph object at 0x12bed5dd0>
print(new_c.graph)  # <tensorflow.python.framework.ops.Graph object at 0x12bed5dd0>

# 指定一个会话运行TensorFlow程序
# 会话的作用是运行一张图，会话也有一张默认图
with tf.Session(graph=new_g) as sess:
    # 会话也有一张对应的默认图，这里与上面的g相同
    # print(sess.graph)  # <tensorflow.python.framework.ops.Graph object at 0x118424b10>
    # sess会话运行默认图和new_c所在的图不同会报错
    # 如果有多个图，需要多个会话开启，指定图的方式：
    # 需要在tf.Session()中指定参数graph=new_g
    c_res = sess.run(new_c)
    print(c_res)


