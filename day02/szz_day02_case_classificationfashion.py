# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/22
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day02_case_classificationfashion.py
构建双层神经网络进行时装模型训练与预测
1. 读取数据集
2. 建立神经网络模型
3. 编译模型优化器，损失，准确率
4. 进行fit训练
5. 评估模型测试效果
"""
import os
import tensorflow as tf
import numpy as np
from tensorflow.python import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SingleNN(object):
    # 2. 建立神经网络模型
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # 将输入数据的性状进行修改成神经网络要求的数据形状
        keras.layers.Dense(128, tf.nn.relu),  # 定义隐藏层，128个神经元的网络层
        # 10个类别的分类问题，输出神经元个数，必须和总类别数相同
        keras.layers.Dense(10, tf.nn.softmax)
    ])

    def __init__(self):
        # 1. 加载数据
        # 获取数据集，返回两个元祖
        # x_train(60000, 784) y_train(60000, 784)
        (self.x_train, self.y_train,), (self.x_test, self.y_test) = keras.datasets.fashion_mnist.load_data()
        # 对于图片数据要对其进行归一化操作（除以255）
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def singlenn_compile(self):
        """
        3. 编译模型优化器、损失、准确率
        :return:
        """
        # 优化器，损失函数
        SingleNN.model.compile(optimizer=keras.optimizers.SGD(lr=0.001),
                               loss=keras.losses.sparse_categorical_crossentropy,
                               metrics=['accuracy'])
        return None

    def singlenn_fit(self):
        """
        进行fit训练
        :return:
        """
        # 在fit当中添加回调函数，记录训练模型过程
        # model_check = keras.callbacks.ModelCheckpoint(
        #     filepath='./ckpt/singlenn_{epoch:02d}.h5',
        #     monitor='val_acc',  # 指定记录损失还是准确率
        #     sava_best_only=True,
        #     save_weights_only=True,
        #     mode='auto',
        #     period=1
        # )

        # 调用tensorboard回调函数
        board = keras.callbacks.TensorBoard(log_dir='./graph/', histogram_freq=1, write_graph=True, write_images=True)
        # 训练样本的特征值和目标值
        SingleNN.model.fit(x=self.x_train, y=self.y_train, epochs=5, batch_size=32,
                           callbacks=[board])

        return None

    def single_evaluate(self):
        # 评估模型测试效果
        test_loss, test_acc = SingleNN.model.evaluate(self.x_test, self.y_test)
        print(test_loss, test_acc)

        return None

    def single_predict(self):
        """
        预测结果
        :return:
        """
        # 直接使用训练过后的权重进行预测
        # if os.path.exists('./ckpt/checkpoint'):
        SingleNN.model.load_weights('./ckpt/SingleNN.h5')
        predictions = SingleNN.model.predict(self.x_test)
        print(predictions)
        return predictions


if __name__ == '__main__':
    snn = SingleNN()
    snn.singlenn_compile()
    snn.singlenn_fit()
    snn.single_evaluate()
    # SingleNN.model.save_weights('./ckpt/SingleNN.h5')

    # 进行模型预测
    # prediction = snn.single_predict()
    # print(prediction)
    # # 求最大值所在的索引值
    # max_value = np.argmax(prediction, axis=1)
    # print(max_value)
