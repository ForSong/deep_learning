# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/19 
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day02.py
"""
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python import keras


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

    (x_train, y_train,), (x_test, y_test), = keras.datasets.cifar100.load_data()
    print(x_train.shape)
    print(y_train.shape)

    (x_train_fa, y_train_fa), (x_test_fa, y_test_fa), = keras.datasets.fashion_mnist.load_data()


if __name__ == '__main__':
    run()
