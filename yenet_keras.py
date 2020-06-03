import tensorflow as tf
import numpy as np


class SRMLayer(tf.keras.Model):
    """
    SRM算子卷积层
    """
    def __init__(self):
        super(SRMLayer, self).__init__()
        self.SRM_kernel = np.load('./SRM_Kernels.npy').astype("float32")

    def call(self, inputs):
        return tf.keras.backend.conv2d(
            inputs, self.SRM_kernel, strides=(1, 1), padding='valid')

class ConvBnPoolLayer(tf.keras.Model):
    """
    卷积，归一化，激活层
    """
    def __init__(self, filters, kernel_size, stride, use_Pool=False, use_TLU=False, TLU_threshold=0, padding='valid'):
        super(ConvBnPoolLayer, self).__init__()
        print("stride", stride)
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                           strides=(stride, stride), padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.use_Pool = use_Pool
        self.use_TLU = use_TLU
        self.TLU_threshold = TLU_threshold
        self.activate = tf.keras.layers.Activation('relu')
        # 如果使用池化层，则初始化一个池化层
        if use_Pool:
            self.pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')


    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if not self.use_TLU:
            # 如果不使用TLU，直接Relu激活
            x = self.activate(x)
        else:
            # 使用TLU，截断在-threshold~threshold之间
            x = tf.keras.activations.relu(x, max_value=self.TLU_threshold,
                                            threshold=-self.TLU_threshold)
        # 如果需要池化操作，则返回池化后的张量，否则直接返回
        if self.use_Pool:
            return self.pool(x)
        else:
            return x

class Yenet(tf.keras.Model):
    def __init__(self, TLU_threshold):
        super(Yenet, self).__init__()
        self.SRMLayer = SRMLayer()
        self.TLULayer1 = ConvBnPoolLayer(30, 5, 1, False, True, TLU_threshold, padding='valid')

        # 从 论文中的layer2开始
        self.LayerList = tf.keras.Sequential()
        self.filterList = [30, 30, 30, 32, 32, 32, 16, 16]
        self.kernelList = [3, 3, 3, 5, 5, 5, 3, 3]
        self.strideList = [1, 1, 1, 1, 1, 1, 1, 3]
        self.poolList = [False, False, True, True, True, True, False, False]
        self.layerNums = len(self.filterList)

        for i in range(self.layerNums):
            print(i)
            self.LayerList.add(ConvBnPoolLayer(self.filterList[i], self.kernelList[i], self.strideList[i], self.poolList[i],
                                               padding='valid'))
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(2, activation=tf.nn.softmax)

    def call(self, inputs):
        # 计算SRM算子得到的概率图
        prob_map = self.SRMLayer(inputs)
        print("SRM张量形状为", prob_map.shape)
        # 计算TLU层
        TLU_tensor = self.TLULayer1(inputs)
        print("TLU张量形状为", TLU_tensor.shape)
        x = tf.keras.layers.add([prob_map, TLU_tensor])
        print("相加后张量形状为", x.shape)
        x = self.LayerList(x)
        print("最终卷积形状为", x.shape)
        x = self.flatten(x)
        x = self.fc(x)

        return x

import os
# 强制使用CPU跑，注意修改下
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":
    model = Yenet(TLU_threshold=3)
    OPT = 'Nadam'
    LOSS = 'categorical_crossentropy'
    model.compile(optimizer=OPT, loss=LOSS, metrics=['accuracy'])
    model.build(input_shape=(None, 256, 256, 1))
    print(model.summary())
