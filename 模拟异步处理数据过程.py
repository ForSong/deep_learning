import tensorflow as tf

# 模拟异步子线程 存入样本， 主线程 读取样本
# 1. 定义一个队列, 1000个
Q = tf.FIFOQueue(1000, tf.float32)

# 2. 定义子线程要做的事情 循环 值 +1 放入队列中
var = tf.Variable(0.0, tf.float32)

# 实现自增 tf.assign_add
data = tf.assign_add(var, tf.constant(1.0))

en_q = Q.enqueue(data)

# 3. 定义队列管理器op，指有多少个子线程，子线程该干什么事情
qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)

# 初始化变量的op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 初始化变量
    sess.run(init_op)
    # 开启线程管理器
    coord = tf.train.Coordinator()

    # 真正的开启子线程
    threads = qr.create_threads(sess, coord=coord, start=True)

    # 主线程，不断的读取数据
    for i in range(300):
        print(sess.run(Q.dequeue()))
    # 注意这里主线程结束，子线程还没有结束，会报错

    # 回收你
    coord.request_stop()

    coord.join(threads)

