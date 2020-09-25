# -*- coding: utf-8 -*-
"""
@Time    : 2020/9/24 
@Author  : Zhizhuo Song
@Email   : zhizhuosong@126.com
@File    : szz_day04.py
迁移学习，使用vgg16模型为基础模型，创建新的全连接层并训练数据，使用训练好的模型处理数据。
"""
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.python.keras.applications.vgg16 import VGG16


class TransferModel():
    def __init__(self):
        # 定义训练和测试图片的变化方式，标准化以及数据增强
        self.train_generator = ImageDataGenerator(rescale=1. / 255)
        self.test_generator = ImageDataGenerator(rescale=1. / 255)

        # 定义图片训练相关的网络参数
        self.train_dir = './data/train'
        self.test_dir = './data/test'

        # 定义图片训练相关网络参数
        self.batch_size = 32
        self.image_size = (224, 224)

        # 获取VGG16基类模型
        self.base_model = VGG16(weights='imagenet', include_top=False)

    def get_local_data(self):
        """
        获取本地数据以及类别
        :return: 训练数据和测试数据迭代器
        """
        # 使用flow_from_directory
        train_gen = self.train_generator.flow_from_directory(self.train_dir
                                                             , target_size=self.image_size
                                                             , batch_size=self.batch_size
                                                             , class_mode='binary'
                                                             , shuffle=True)
        test_gen = self.test_generator.flow_from_directory(self.test_dir
                                                           , target_size=self.image_size
                                                           , batch_size=self.batch_size
                                                           , class_mode='binary'
                                                           , shuffle=True)

        return train_gen, test_gen

    def refine_base_model(self):
        """
        微调vgg16模型结构，5blocks后面加全局池化（减少迁移学习的参数数量），增加两个全连接层
        :return:
        """
        # 1. 获取notop模型，[?,?,?,512]
        x = self.base_model.outputs[0]
        # 全局池化
        x = keras.layers.GlobalAveragePooling2D()(x)
        # 2. 在输出之后增加自己的神经网络层
        x = keras.layers.Dense(1024, activations=tf.nn.relu)(x)
        y_predict = keras.layers.Dense(5, activations=tf.nn.softmax)(x)

        # 定义新的模型
        new_model = keras.models.Model(inputs=self.base_model.inputs, outputs=y_predict)

        return new_model

    def freeze_model(self):
        """
        冻结VGG模型，需要冻结的比例根据实际的需要选择
        :return:无返回值
        """
        for layer in self.base_model.layers:
            layer.trainable = False

    def compile_model(self, model):
        model.compile(optimizer=keras.optimizers.Adam()
                      , loss=keras.losses.sparse_categorical_crossentropy
                      , metrics=['accuracy']
                      )

        return None

    def fit_generator(self, model, train_gen, test_gen):
        """
        用数据fit模型
        :return:
        """
        model_ckpt = keras.callbacks.ModelCheckpoint('./ckpt/transfer_{epochs:02d}-{val_acc:.2f}.h5'
                                                     , monitor='val_acc'
                                                     , save_weights_only=True
                                                     , save_best_only=True
                                                     , mode='auto'
                                                     , period=1)
        model.fit_generator(train_gen, epochs=3, validation_data=test_gen, callbacks=[model_ckpt])

        return None

    def train_model(self):
        """
        训练模型
        :return:
        """
        # 1. 获取本地数据和类别
        train_gen, test_gen = self.get_local_data()
        # 2. 导入VGG16并微调结构
        new_model = self.refine_base_model()
        # 3. 冻结模型
        self.freeze_model()
        # 4. 编译模型
        self.compile_model(new_model)
        # 5. 训练模型
        self.fit_generator(new_model, train_gen, test_gen)
        # return new_model

    def load_model(self, model):
        # TODO,训练好后，将模型文件的路径填入
        model.load_weights('模型文件路径')
        return model

    def process_picture(self, path):
        # 使用keras.preprocessing.image接口中的方法加载图片
        image = load_img(path)
        image = img_to_array(image)

        # 对图片进行维度的转换，这里需要四维的数据
        img = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
        return img


    def predict(self, path):
        # 1. 加载模型
        model = self.refine_base_model()
        model = self.load_model(model)

        # 2. 加载图片并进行处理
        img = self.process_picture(path)
        predict_res = model.predict(img)

        return predict_res


if __name__ == '__main__':
    tm = TransferModel()
    # part1，训练模型， 只运行一次，得到文件即可
    tm.train_model()
    # part2, 预测结果
    # TODO 填入要预测的文件路径即可
    print(tm.predict('要预测的文件的路径'))
