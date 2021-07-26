from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, DepthwiseConv2D, Reshape, Concatenate, \
    Input, BatchNormalization, Activation, Conv2D, InputLayer, UpSampling1D, AveragePooling2D, LayerNormalization, \
    Conv1D, SpatialDropout2D
from tensorflow.keras.regularizers import l1
from tensorflow.keras import backend as K


def depthwise_model_ndms(input_shape, output_shape,
                         depth_mul_in=(4, 4), filters_out=(4, 4),
                         krnl_in=((1, 3), (1, 3)), krnl_out=(3, 3),
                         pad='valid', strides=((1, 1), (1, 1)),
                         dil=((1, 1), (1, 1)),
                         mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu', 'relu'),
                         feature_conv=None,
                         b_norm=False, l_norm=False,
                         dense_drp=False, conv_drp=False,  drp=0.3):
    model = Sequential(name='Depthwise_model')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer(input_shape=input_shape))

    # model.add(AveragePooling2D(pool_size=(1,2), strides=(1, 2)))

    for i in range(len(krnl_in)):
        model.add(DepthwiseConv2D(kernel_size=krnl_in[i],
                                  depth_multiplier=depth_mul_in[i],
                                  activation=acts[0],
                                  padding=pad,
                                  strides=strides[i],
                                  dilation_rate=dil[i]))
        if mpool[i][0]:
            model.add(MaxPooling2D(pool_size=mpool[i]))
        if l_norm:
            model.add(LayerNormalization(axis=-1))
        if conv_drp:
            model.add(SpatialDropout2D(drp/4))

    if feature_conv is not None:
        model.add(Conv2D(kernel_size=(1, 1), filters=feature_conv, activation=acts[0], padding='same',))
        model.add(SpatialDropout2D(drp/4))

    model.add(Flatten())

    if b_norm:
        for i, d in enumerate(dense):
            model.add(Dense(d))
            model.add(Activation(acts[1]))
            model.add(BatchNormalization())

            if dense_drp and i != len(dense)-1:
                model.add(Dropout(drp))
    else:
        for i, d in enumerate(dense):
            model.add(Dense(d, activation=acts[1]))
            if dense_drp and i != len(dense)-1:
                model.add(Dropout(drp))

    model.add(Reshape((15, -1)))

    for i in range(len(krnl_out)):
        model.add(UpSampling1D(size=2))
        model.add(Conv1D(filters=filters_out[i], kernel_size=krnl_out[i], padding='same', activation=acts[2]))
        if l_norm:
            model.add(LayerNormalization(axis=-1))

    model.add(Conv1D(kernel_size=1, filters=2, padding='same', activation='linear'))

    model.compile(loss=MeanSquaredError(), optimizer=Adam())  # metrics=[coeff_determination]
    return model


def depthwise_model_ndms_encode(input_shape, output_shape,
                                 depth_mul_in=(4, 4),
                                 krnl_in=((1, 3), (1, 3)),
                                 pad='valid', strides=((1, 1), (1, 1)),
                                 dil=((1, 1), (1, 1)),
                                 mpool=((0, 0), (0, 0)), dense=(100, 50), acts=('relu', 'relu', 'relu'),
                                 feature_conv=None,
                                 b_norm=False, l_norm=False,
                                 dense_drp=False, conv_drp=False, drp=0.3):

    model = Sequential(name='Depthwise_model')
    if len(input_shape) < 3:
        model.add(Reshape((1, *input_shape), input_shape=input_shape))
    else:
        model.add(InputLayer(input_shape=input_shape))

    # model.add(AveragePooling2D(pool_size=(1,2), strides=(1, 2)))

    for i in range(len(krnl_in)):
        model.add(DepthwiseConv2D(kernel_size=krnl_in[i],
                                  depth_multiplier=depth_mul_in[i],
                                  activation=acts[0],
                                  padding=pad,
                                  strides=strides[i],
                                  dilation_rate=dil[i]))
        if mpool[i][0]:
            model.add(MaxPooling2D(pool_size=mpool[i]))
        if l_norm:
            model.add(LayerNormalization(axis=-1))
        if conv_drp:
            model.add(SpatialDropout2D(drp/4))

    if feature_conv is not None:
        model.add(Conv2D(kernel_size=(1, 1), filters=feature_conv, activation=acts[0], padding='same',))
        model.add(SpatialDropout2D(drp/4))

    model.add(SpatialDropout2D(drp / 2))
    model.add(Flatten())

    if b_norm:
        for i, d in enumerate(dense):
            model.add(Dense(d))
            model.add(Activation(acts[1]))
            model.add(BatchNormalization())

            if dense_drp and i != len(dense)-1:
                model.add(Dropout(drp))
    else:
        for i, d in enumerate(dense):
            model.add(Dense(d, activation=acts[1]))
            if dense_drp and i != len(dense)-1:
                model.add(Dropout(drp))
    model.add(Dense(np.product(output_shape), activation='linear'))
    model.add(Reshape(output_shape))

    model.compile(loss=MeanSquaredError(), optimizer=Adam())  # metrics=[coeff_determination]
    return model
