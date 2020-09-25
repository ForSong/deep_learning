# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/23 
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day03.py
卷积神经网络的建立
1. 读取数据集
2. 编写两层+两层全连接层网络模型
3. 编译，训练，评估
第一层：
卷积：[None,32,32,3]--->（32个卷积核）[None,32,32,32]
权重数量：[5,5,3,32]
偏置数量：[32]
激活：[None,32,32,32]--->[None,32,32,32]
池化：[None,32,32,32]--->[None,16,16,32]
第二层：
卷积：[None,16,16,32]--->[None,16,16,64]
权重数量：[5,5,32,64]
偏置数量：[64]
激活：[None,16,16,64]--->[None,16,16,64]
池化：[None,16,16,64]--->[None,8,8,64]
全连接层：
[None,8,8,64]--->[None, 8*8*64]
[None,8 8 64] x [8 8 64, 1024] = [None,1024]
[None,1024]x[1024,100]--->[None, 100]
权重数量：[8 8 64, 1024] + [1024,100],由分类数量决定
偏置数量：[1024] + [100]由分类数量决定
"""
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar100
from tensorflow.python import keras


class CNNMnist(object):
    # 2. 编写两层+两层全连接网络模型
    model = keras.models.Sequential([
        # 卷积层1： 32个 5*5*3的filter， strides=1,padding="same"
        keras.layers.Conv2D(32, kernel_size=5, strides=1, padding='same', data_format='channels_last',
                            activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.01)),
        # 池化层1：2*2窗口,strides=2
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
        # 卷积层2：64个 5*5*32个filter，strides=1,padding='same'
        keras.layers.Conv2D(64, kernel_size=5, strides=1, padding='same', data_format='channels_last',
                            activation=tf.nn.relu),
        # 池化层2 输出：[None,8,8,64]
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
        # 全连接层神经网络： 将之前的数据展平：[None,8,8,64]--->[None,8*8*64]
        keras.layers.Flatten(),
        # 1024个神经元网络层
        keras.layers.Dense(1024, activation=tf.nn.relu),
        # 100 个神经元网络层
        keras.layers.Dense(100, activation=tf.nn.softmax)
    ])

    def __init__(self):
        # 1. 获取训练测试数据
        (self.x_train, self.y_train), (self.x_test, self.y_test), = cifar100.load_data()
        print(self.x_train.shape)
        print(self.x_test.shape)

        # 进行数据归一化
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def compile(self):
        CNNMnist.model.compile(optimizer=keras.optimizers.Adam(),
                               loss=tf.keras.losses.sparse_categorical_crossentropy,
                               metrics=['accuracy'])
        return None

    def fit(self):
        CNNMnist.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return None

    def evaluate(self):
        test_loss, test_acc = CNNMnist.model.evaluate(self.x_test, self.y_test)
        print(test_loss, test_acc)


if __name__ == '__main__':
    cnn_mnist = CNNMnist()
    # print(CNNMnist.model.summary())
    cnn_mnist.compile()
    cnn_mnist.fit()
    cnn_mnist.evaluate()
    print(cnn_mnist.model.summary())
