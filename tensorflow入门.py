import tensorflow as tf
# 消除警告
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建一张图包含了一组op和tensor, 使用上下文环境
# op：只要使用tensorflow的API定义的函数都是op
# tensor(张量):就指代的是数据
# 在计算的时候使用的tensor计算,op相当于是一个载体,tensor相当与是一个被载的数据

# g = tf.Graph()
#
# print(g)
# with g.as_default():
#     c = tf.constant(11.0)
#     print(c.graph)
#
# # 实现一个加法运算
#
# a = tf.constant(5.0)
# b = tf.constant(6.0)
#
# sum1 = tf.add(a, b)
#
# # 默认的定义图对象，主要作用是分配一段内存
# graph = tf.get_default_graph()
# print(graph)
#
# # 如果定义的不是op不能运行，下面是定义成了整型变量
# var1 = 2
# # var2 = 4
# # sum2 = var1 + var2
#
# # 有重载机制
# # 只要有一部分是op类型，那么默认就会将运算符重载成op类型
# sum2 = a + var1
# print(sum2)
# # print(sum1)
#
# # 只能运行一个图
# # 但是可以在会话中指定图去运行
#
# # 训练模型，
# # 实时的提供数据进行训练
#
# # placeholder是一个占位符,这里也是一个op
# plt = tf.placeholder(tf.float32, [None, 3])
#
# print(plt)
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     print(sess.run(plt, feed_dict={plt: [[1, 2, 3], [4, 5, 6], [2, 3, 4]]}))
#     # print(sum1.eval())
#     # 只要程序定义好之后，程序内部的所有内容都在同一个地方
#     print(a.graph)
#     print(sum1.graph)
#     print(sess.graph)
#
# # tensorflow分为前端和后端，前端用于定义程序的图的机构，后端用于计算图的结构
# # 会话：1. 运行图的结构；2.分配资源计算 3. 掌握资源(变量，队列，线程)
# # sess.run()相当于启动
# # 形状改变 reshape
# 对于静态形状来说，一旦张量形状固定了，就不能再次设置静态形状，而且不能跨维度修改
# 动态形状可以创建一个新的张量，可以跨纬度改变，但是一定要保持数量与之前的一致

# plt = tf.placeholder(tf.float32, [None, 2])
#
# print(plt)
#
# plt.set_shape([3, 2])
#
# print(plt)
#
# # plt.set_shape([4, 2])
# # 动态形状修改，但是元素的数量要和之前的数量保持一致，
# plt_reshape = tf.reshape(plt, [2, 3])
#
# print(plt_reshape)
#
# with tf.Session() as sess:
#     pass


# 变量op
# 1. 变量op能够持久化保存，普通的张量op不行
# 2. 当定义一个变量op的时候，一定要在会话中初始化（运行）
# 3. name参数：在tensorboard使用的时候显示名字，可以让相同op名字的进行区分

a = tf.constant(3.0, name="a")
b = tf.constant(4.0, name="b")
c = tf.add(a, b, name="add")
# 三个参数，第一个是形状，第二个是平均值，第三个是标准差
var = tf.Variable(tf.random_normal([2, 4], mean=0.0, stddev=1.0), name="variable")
print(a, var)
# 必须做一步显示的初始化
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 必须运行初始化op
    sess.run(init_op)
    # 把程序的图结构写入事件文件,把指定的图写进事件文件中
    file_writer = tf.summary.FileWriter("./summary/test/", graph=sess.graph)
    print(sess.run([c, var]))
