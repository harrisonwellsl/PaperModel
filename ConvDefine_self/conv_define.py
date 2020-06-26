from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers import LeakyReLU, BatchNormalization, MaxPooling2D, AveragePooling2D


class ConvDefine:
    # 卷积操作
    # 参数axis只有在use_batch_normalization参数设置为True的时候才起作用
    # 如果不开启use_batch_normalization，那么即使设置了axis也不起作用
    # if use_batch_normalization:
    #     inpt = BatchNormalization(axis=axis)(inpt)
    @staticmethod
    def self_conv2d(inpt, filter_num=64,
                    kernal_size=(3, 3), strides=(1, 1),
                    padding='same', activation=None,
                    use_leaky_relu=False, alpha=0.01, data_format='channels_last'):
        if activation is not None:
            inpt = Conv2D(filters=filter_num, kernel_size=kernal_size, strides=strides,
                          padding=padding, activation=activation, data_format=data_format)(inpt)
        elif use_leaky_relu and activation is None:
            inpt = Conv2D(filters=filter_num, kernel_size=kernal_size, strides=strides,
                          padding=padding, activation=activation, data_format=data_format)(inpt)
            inpt = LeakyReLU(alpha=alpha)(inpt)
        else:
            raise ValueError
        return inpt

    # 反卷积操作
    # 参数axis只有在use_batch_normalization参数设置为True的时候才起作用
    # 如果不开启use_batch_normalization，那么即使设置了axis也不起作用
    @staticmethod
    def self_conv2d_t(inpt, filter_num=64,
                    kernal_size=(2, 2), strides=(2, 2),
                    padding='same', activation=None,
                    use_leaky_relu=False, alpha=0.01, data_format='channels_last'):
        if activation is not None:
            inpt = Conv2DTranspose(filters=filter_num, kernel_size=kernal_size, strides=strides,
                          padding=padding, activation=activation, data_format=data_format)(inpt)
        elif use_leaky_relu and activation is None:
            inpt = Conv2DTranspose(filters=filter_num, kernel_size=kernal_size, strides=strides,
                          padding=padding, activation=activation, data_format=data_format)(inpt)
            inpt = LeakyReLU(alpha=alpha)(inpt)
        else:
            raise ValueError
        return inpt

    # 池化操作
    # 单独封装为一个函数，如果有可以优化的可以直接添加
    @staticmethod
    def self_avrage_pooling(data_input, pool_size=(2, 2),
                         strides=None, padding='same',
                         data_format='channels_last'):
        output = AveragePooling2D(pool_size=pool_size, data_format=data_format,
                                  strides=strides, padding=padding)(data_input)
        return output