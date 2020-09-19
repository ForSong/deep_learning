import tensorflow as tf

# 模拟同步：先处理数据，然后才取出数据
# tensorflow中，运行操作有依赖性，也就是说运行最后一步，存在依赖的步骤就会运行

# 一。先定义图的结构
# 1. 首先定义队列
Q = tf.FIFOQueue(3, tf.float32)
# 2. 定义一些读取数据，取数据的过程：
#       取数据 +1
#       入队列

# 将数据放入队列
# 这里的参数要注意，如果直接用一个列表会报错，因为会被识别为一个张量
enq_many = Q.enqueue_many([[0.1, 0.2, 0.3],])
# 取出数据
out_q_data = Q.dequeue()
# 数据+1这里重载了
data = out_q_data + 1
# 计算结果入队
en_q = Q.enqueue(data)


with tf.Session() as sess:
    # 初始化队列
    sess.run(enq_many)
    # 处理数据
    for i in range(100):
        sess.run(en_q)
    # 读取数据，注意：Q.size()是一个op，因此需要使用eval()运行
    for i in range(Q.size().eval()):
        run = sess.run(Q.dequeue())
        print(run)
