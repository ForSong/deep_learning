import tensorflow as tf
# 消除警告
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 训练参数，在Variable变量类型中，有一个参数trainable如果设置为True那么就会不断的优化，如果是False
# 就不会变化，相当于constant。

# 学习率：一般设置比较小，一般为两位或者三位小数，0.01，0.001，0.0001
# 学习率越小，那么所需要的训练次数越多。
# 梯度爆炸：当学习率很大的时候，权重(weight)值变的非常大，以至于溢出，导致出现NaN值
# 产生的原因是权重产生指数型变化。
# 如何解决梯度爆炸？1. 调整学习率；2.

# 添加权重参数，损失值等在tensorboard观察的情况步骤：1. 收集变量 2. 合并变量写入时间文件

def my_regression():
    """
    用tensorflow中的接口实现线性回归
    :return: None
    """
    with tf.variable_scope("data"):
        # 1. 准备数据， x特征值，[100,1] 100行1列，y目标值[100]
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")
        # 矩阵相乘必须是二维的,这里使用随机生成的x，给定的w=0.7 b=0.8 去计算真实的y值
        y_true = tf.matmul(x, [[0.7]]) + 0.8
    with tf.variable_scope("model"):
        # 2. 建立线性回归模型 1 个特征 1个权重，1个偏置 y = x * w + b
        # 随即指定初始的权重和偏置，计算损失并优化
        # 用变量定义才能优化
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0, name="w"))
        bias = tf.Variable(0.0, name="b")

        y_predict = tf.matmul(x, weight) + bias
    with tf.variable_scope("loss"):
        # 3. 建立损失函数，均方误差
        loss = tf.reduce_mean(tf.square(y_true - y_predict))
    with tf.variable_scope("optimizer"):
        # 4. 梯度下降优化损失, 参数为学习率
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 定义一个初始化变量op
    init_op = tf.global_variables_initializer()

    # 如何在tensorboard中显示变量的变换情况
    # 1. 在这里收集tensor，用来在tensorboard中显示
    tf.summary.scalar("losses", loss)

    tf.summary.histogram("weights", weight)
    # 2. 合并变量并写入时间文件,返回一个op
    merged = tf.summary.merge_all()

    # 通过会话运行程序
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)
        # 打印随机最先初始化的权重和偏置
        print("随机初始化的参数权重为： %f, 偏置为: %f" % (weight.eval(), bias.eval()))
        # 建立时间文件
        fileWriter = tf.summary.FileWriter("./summary/test", graph=sess.graph)

        # 循环训练运行优化
        for i in range(300):
            sess.run(train_op)
            # 运行合并的tensor
            summary = sess.run(merged)
            fileWriter.add_summary(summary, i)
            print("参数权重为：%f，偏置为：%f" % (weight.eval(), bias.eval()))


if __name__ == '__main__':
    my_regression()
