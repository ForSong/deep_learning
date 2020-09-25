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
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Flatten, Input
import tensorflow as tf
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python import keras
import os
import numpy as np


# def main():
#
#     image = load_img("./bus/300.jpg", target_size=(300, 300))
#     #
#     # print(image)
#     # # 输入到tensorflow，做处理
#     # image = img_to_array(image)
#     # print(image.shape)
#     # print(image)
#
#     # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
#     # print(x_train.shape)
#     # print(y_train.shape)
#
#     # (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
#     # print(x_train)
#     # print(y_train)
#
#     # (784, )---->(28, 28)
#     model_first = Sequential([
#         Flatten(input_shape=(28, 28)),
#         Dense(64, activation=tf.nn.relu),
#         Dense(128, activation=tf.nn.relu),
#         Dense(10, activation=tf.nn.softmax)
#     ])
#
#     # 通过Model建立模型
#     data = Input(shape=(784, ))
#     print(data)
#     out = Dense(64)(data)
#     print(out)
#     model_sec = Model(inputs=data, outputs=out)
#     print(model_sec)
#     print(model_first.layers, model_sec.layers)
#     print(model_first.inputs, model_first.outputs)
#
#     # 模型结构参数
#     print(model_first.summary())
#
#
# if __name__ == '__main__':
#     main()

# 构建双层神经网络去进行时装模型训练与预测
#   -1、读取数据集
# # #   - 2、建立神经网络模型
# # #   - 3、编译模型优化器、损失、准确率
# # #   - 4、进行fit训练
# # #   - 评估模型测试效果


class SingleNN(object):
    # 建立神经网络模型
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # 将输入数据的形状进行修改成神经网络要求的数据形状
        keras.layers.Dense(128, activation=tf.nn.relu),  # 定义隐藏层，128个神经元的网络层
        keras.layers.Dense(10, activation=tf.nn.softmax)  # 10个类别的分类问题，输出神经元个数必须跟总类别数量相同
    ])

    def __init__(self):
        # 返回两个元组
        # x_train: (60000, 784), y_train:(60000, 1)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.fashion_mnist.load_data()

        # 进行数据的归一化
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def singlenn_compile(self):
        """
        编译模型优化器、损失、准确率
        :return:
        """
        # 优化器
        # 损失函数
        SingleNN.model.compile(optimizer=keras.optimizers.Adam(),
                               loss=keras.losses.sparse_categorical_crossentropy,
                               metrics=['accuracy'])

        return None

    def singlenn_fit(self):
        """
        进行fit训练
        :return:
        """
        # # fit当中添加回调函数，记录训练模型过程
        # modelcheck = keras.callbacks.ModelCheckpoint(
        #     filepath='./ckpt/singlenn_{epoch:02d}-{val_loss:.2f}.h5',
        #     monitor='val_loss',  # 保存损失还是准确率
        #     save_best_only=True,
        #     save_weights_only=True,
        #     mode='auto',
        #     period=1
        # )
        # 调用tensorboard回调函数
        board = keras.callbacks.TensorBoard(log_dir="./graph/", write_graph=True)

        # 训练样本的特征值和目标值
        SingleNN.model.fit(self.x_train, self.y_train, epochs=5,
                           batch_size=128, callbacks=[board])

        return None

    def single_evalute(self):
        # 评估模型测试效果
        test_loss, test_acc = SingleNN.model.evaluate(self.x_test, self.y_test)

        print(test_loss, test_acc)

        return None

    def single_predict(self):
        """
        预测结果
        :return:
        """
        # 首先加载模型
        # if os.path.exists("./ckpt/checkpoint"):
        SingleNN.model.load_weights("./ckpt/SingleNN.h5")

        predictions = SingleNN.model.predict(self.x_test)

        return predictions


if __name__ == '__main__':
    snn = SingleNN()

    snn.singlenn_compile()

    snn.singlenn_fit()

    snn.single_evalute()

    # SingleNN.model.save_weights("./ckpt/SingleNN.h5")
    # 进行模型预测
    # predictions = snn.single_predict()
    # # [10000, 10]
    # print(predictions)
    # print(np.argmax(predictions, axis=1))
