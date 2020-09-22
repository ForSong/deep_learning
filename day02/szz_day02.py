# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/19 
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day02.py
"""
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Flatten, Input


def run():
    # part 1 start 读取图片并设置大小
    # img = load_img('./bus/300.jpg', target_size=(300, 300))
    # print(img)
    # # return img
    # # 输入到tensorflow做处理，需要转换类型
    # image = img_to_array(img)
    # print(image.shape)
    # print(image)
    # part 1 end

    # (x_train, y_train,), (x_test, y_test), = keras.datasets.cifar100.load_data()
    # print(x_train.shape)
    # print(y_train.shape)
    #
    # (x_train_fa, y_train_fa), (x_test_fa, y_test_fa), = keras.datasets.fashion_mnist.load_data()
    # 通过Sequential建立模型
    model_fir = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.softmax)
    ])
    print(model_fir)

    # 通过model建立模型
    data = Input(shape=(784,))
    print(data)
    out = Dense(64)(data)
    print(out)
    model_sec = Model(inputs=data, outputs=out)
    print(model_sec)
    print(model_fir.layers, model_sec.layers)
    print(model_fir.inputs, model_fir.outputs)

    # 模型结构参数
    print(model_fir.summary())


if __name__ == '__main__':
    run()
