# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/23 
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day03_VGG.py
调用vgg16进行图像分类。
如果想要识别的类别在VGG中已经存在，就不需要再自己建立神经网络
可以使用已有的模型进行直接预测即可
1. 获取模型以及训练好的参数
"""
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


def predict():
    model = VGG16()
    # print(model.summary())
    # 1，加载图片
    image = load_img('./tiger.jpg', target_size=(224, 224))
    image = img_to_array(image)
    print(image.shape)

    # 2. 转换数据，使其满足卷积神经网络的要求, 卷积神经网络要求的输入是4个维度，第一个维度是
    # 输入图片的数量
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    # 使用接口对数据进行归一化等处理
    image = preprocess_input(image)
    # 处理完成之后开始进行预测
    prediction = model.predict(image)
    # 打印预测结果
    print(prediction)
    # 对预测结果进行解码
    label = decode_predictions(prediction)
    # 打印解码后的结构
    print(label)
    # 根据解码结果，取出想要的内容
    print("预测的结果是：{}, 预测的概率是：{}".format(label[0][0][1], label[0][0][2]))


if __name__ == '__main__':
    predict()
