from keras import losses
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D
from keras.layers import Dropout
from ConvDefine_self.conv_define import ConvDefine


def unet_self(shape: tuple, loss_function=None, optimizer_self=None, metrics_self=None):
    inp = Input(shape=shape)

    conv1 = ConvDefine.self_conv2d(inp, filter_num=8, use_leaky_relu=True)
    conv1 = ConvDefine.self_conv2d_t(conv1, filter_num=8, use_leaky_relu=True)
    pooling1 = ConvDefine.self_avrage_pooling(conv1)

    conv2 = ConvDefine.self_conv2d(pooling1, filter_num=16, use_leaky_relu=True)
    conv2 = ConvDefine.self_conv2d(conv2, filter_num=16, use_leaky_relu=True)
    pooling2 = ConvDefine.self_avrage_pooling(conv2)

    conv3 = ConvDefine.self_conv2d(pooling2, filter_num=32, use_leaky_relu=True)
    conv3 = ConvDefine.self_conv2d(conv3, filter_num=32, use_leaky_relu=True)
    pooling3 = ConvDefine.self_avrage_pooling(conv3)

    conv4 = ConvDefine.self_conv2d(pooling3, filter_num=64, use_leaky_relu=True)
    conv4 = ConvDefine.self_conv2d(conv4, filter_num=64, use_leaky_relu=True)
    pooling4 = ConvDefine.self_avrage_pooling(conv4)

    conv5 = ConvDefine.self_conv2d(pooling4, filter_num=128, use_leaky_relu=True)
    # conv5 = Dropout(0.5)(conv5)
    conv5 = ConvDefine.self_conv2d(conv5, filter_num=128, use_leaky_relu=True)
    # conv5 = Dropout(0.5)(conv5)

    conv_t_1 = ConvDefine.self_conv2d_t(conv5, filter_num=64, use_leaky_relu=True)
    # connect1 = concatenate([conv_t_1, conv4], axis=3)
    connect1 = concatenate([conv_t_1, conv4])
    # connect1 = Dropout(0.5)(connect1)
    conv6 = ConvDefine.self_conv2d(connect1, filter_num=64, use_leaky_relu=True)
    conv6 = ConvDefine.self_conv2d(conv6, filter_num=64, use_leaky_relu=True)

    conv_t_2 = ConvDefine.self_conv2d_t(conv6, filter_num=32, use_leaky_relu=True)
    # connect2 = concatenate([conv_t_2, conv3], axis=3)
    connect2 = concatenate([conv_t_2, conv3])
    # connect2 = Dropout(0.5)(connect2)
    conv7 = ConvDefine.self_conv2d(connect2, filter_num=32, use_leaky_relu=True)
    conv7 = ConvDefine.self_conv2d(conv7, filter_num=32, use_leaky_relu=True)

    conv_t_3 = ConvDefine.self_conv2d_t(conv7, filter_num=16, use_leaky_relu=True)
    # connect3 = concatenate([conv_t_3, conv2], axis=3)
    connect3 = concatenate([conv_t_3, conv2])
    # connect3 = Dropout(0.5)(connect3)
    conv8 = ConvDefine.self_conv2d(connect3, filter_num=16, use_leaky_relu=True)
    conv8 = ConvDefine.self_conv2d(conv8, filter_num=16, use_leaky_relu=True)

    conv_t_4 = ConvDefine.self_conv2d_t(conv8, filter_num=8, use_leaky_relu=True)
    # connect4 = concatenate([conv_t_4, conv1], axis=3)
    connect4 = concatenate([conv_t_4, conv1])
    # connect4 = Dropout(0.5)(connect4)
    conv9 = ConvDefine.self_conv2d(connect4, filter_num=8, use_leaky_relu=True)
    conv9 = ConvDefine.self_conv2d(conv9, filter_num=8, use_leaky_relu=True)
    # conv9 = Dropout(0.5)(conv9)
    # output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1))(conv9)
    output = ConvDefine.self_conv2d(conv9, filter_num=shape[2], kernal_size=(1, 1), strides=(2, 2), activation="linear")

    model = Model(inputs=inp, outputs=output)
    if loss_function is not None and optimizer_self is not None and metrics_self is not None:
        model.compile(loss=loss_function, optimizer=optimizer_self, metrics=metrics_self)
        model.summary()
    else:
        model.compile(loss=['MSE'], optimizer='rmsprop', metrics=['mae'])
        model.summary()
    return model
