# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/14 
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day01_regression.py
"""
import tensorflow as tf


# 命令行参数
# tf.app.flags.DEFINE_integer('max_step', 1000, 'train step number')
# 
# FLAGS = tf.app.flags.FLAGS

def linear_regression():
    # 1. 构造数据,100行1列的数据
    with tf.variable_scope('original_data'):
        x = tf.random_normal([100, 1], mean=0.0, stddev=1.0, name='original_data_x')
        y_true = tf.matmul(x, [[0.8]]) + 0.7

    # 2. 构造模型
    with tf.variable_scope('linear_model'):
        w = tf.Variable(initial_value=tf.random_normal([1, 1]), name='w')
        b = tf.Variable(initial_value=tf.random_normal([1]), name='b')
        y_predict = tf.matmul(x, w) + b

    # 3. 构造损失函数
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 4. 利用梯度下降求解权重和偏置
    with tf.variable_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # 收集要观察的张量，scalar低纬度，histogram高纬度
    tf.summary.scalar('error', loss)
    tf.summary.histogram('weights', w)
    tf.summary.histogram('bias', b)

    # 合并变量
    merge = tf.summary.merge_all()
    # 初始化变量
    init_op = tf.global_variables_initializer()

    # 创建一个saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        writer = tf.summary.FileWriter('./tmp/summary', graph=sess.graph)
        # print('1')
        for i in range(1000):
            sess.run(optimizer)
            # print('loss:', sess.run(loss))
            # print('weight:', sess.run(w))
            # print('bias:', sess.run(b))
            summary_res = sess.run(merge)
            writer.add_summary(summary_res, i)
            # checkpoint:检查点文件格式
            # tf.keras: h5
            saver.save(sess, './tmp/ckpt/linear_regression')

        # 读取模型
        # saver.restore(sess, './tmp/ckpt/linear_regression')
        # print(w.eval())


if __name__ == '__main__':
    linear_regression()
