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

# 定义命令行参数
# 1. 首先定义有哪些参数需要在命令行运行时指定
# 2. 程序当中获取定义命令行参数

# 定义需要指定的参数
# 对于这个方法的参数，第一个参数是方法的名字，第二个参数是方法的默认值，第三个参数是方法的说明
tf.app.flags.DEFINE_integer("max_step", 100, "模型训练的步数")
tf.app.flags.DEFINE_string("model_dir", "", "模型文件的加载路径")

# 定义获取命令行参数的名字
FLAGS = tf.app.flags.FLAGS


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

    # 定义一个保存模型的实例
    saver = tf.train.Saver()

    # 通过会话运行程序
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)
        # 打印随机最先初始化的权重和偏置
        print("随机初始化的参数权重为： %f, 偏置为: %f" % (weight.eval(), bias.eval()))
        # 建立时间文件
        fileWriter = tf.summary.FileWriter("./summary/test", graph=sess.graph)
        # 加载模型，覆盖模型当中随机定义的参数，从上次训练的参数结果开始
        # 首先判断是否有已经输出的模型文件
        if os.path.exists('./model/regression'):
            saver.restore(sess, './model/regression')
        # 循环训练运行优化
        for i in range(FLAGS.max_step):
            sess.run(train_op)
            # 运行合并的tensor
            summary = sess.run(merged)
            fileWriter.add_summary(summary, i)
            print("参数权重为：%f，偏置为：%f" % (weight.eval(), bias.eval()))
            # 如果循环次数比较多的时候可以使用判断语句来保存模型
            # 比如每100次运算保存一次模型
        # 如果运算次数比较少的情况下，可以直接在模型运算完之后保存
        # 注意要首先定义op，记住op永远是在session之前定义的
        # checkpoint 文件是变量名称
        # 第二个是保存的内容
        saver.save(sess, FLAGS.model_dir)


if __name__ == '__main__':
    my_regression()
