"""Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.

The model is introduced in:

Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
Joao Carreira, Andrew Zisserman
https://arxiv.org/abs/1705.07750v1
"""

from __future__ import print_function
from __future__ import absolute_import

import warnings

import numpy as np

from keras.models import Model, load_model
from keras import layers
from keras.layers import (Activation, Dense, Input, BatchNormalization,
                          Conv3D, MaxPooling3D, Dropout, Reshape, Lambda, AveragePooling3D)
from keras.utils.data_utils import get_file
from keras import backend as K
import os

WEIGHTS_NAME = ['rgb_kinetics_only',
                'flow_kinetics_only',
                'rgb_imagenet_and_kinetics',
                'flow_imagenet_and_kinetics']

# path to pretrained models with no top (no classification layer)
WEIGHTS_PATH_NO_TOP = {
    'rgb_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'rgb_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
}


def conv3d_bn(x,
              filters,
              num_frames,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1, 1),
              use_bias=False,
              use_activation_fn=True,
              use_bn=True,
              name=None):
    """Utility function to apply conv3d + BN.
    # Arguments
        x: input.
        filters: filters in `Conv3D`.
        num_frames: frames (time depth) of the convolution kernel.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        use_bias: use bias or not
        use_activation_fn: use an activation function or not.
        use_bn: use batch normalization or not.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv3D(
        filters, (num_frames, num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=conv_name)(x)

    if use_bn:
        bn_axis = 4
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if use_activation_fn:
        x = Activation('relu', name=name)(x)

    return x


def Inception_Inflated3d(include_top=True,
                         weights=None,
                         input_tensor=None,
                         input_shape=None,
                         dropout_prob=0.0,
                         endpoint_logit=True,
                         classes=400):
    """Instantiates the Inflated 3D Inception architecture.
    Optionally loads weights pre-trained on Kinetics.
    Note that when using TensorFlow, for best performance you should set
    `image_data_format='channels_last'` in your Keras config at ~/.keras/keras.json.
    # Arguments
        include_top: whether to include the the classification
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or 'kinetics_only' (pre-training on Kinetics dataset only).
            or 'imagenet_and_kinetics' (pre-training on ImageNet and Kinetics datasets).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(NUM_FRAMES, 224, 224, 3)`
            NUM_FRAMES should be no smaller than 8. The authors used 64
            frames per example for training and testing on kinetics dataset
            Also, Width and height should be no smaller than 32.
            E.g. `(64, 150, 150, 3)` would be one valid value.
        dropout_prob: optional, dropout probability applied in dropout layer
            after global average pooling layer.
            0.0 means no dropout is applied, 1.0 means dropout is applied to all features.
            Note: Since Dropout is applied just before the classification
            layer, it is only useful when `include_top` is set to True.
        endpoint_logit: (boolean) optional. If True, the model's forward pass
            will end at producing logits. Otherwise, softmax is applied after producing
            the logits to produce the class probabilities prediction. Setting this parameter
            to True is particularly useful when you want to combine results of rgb model
            and optical flow model.
            - `True` end model forward pass at logit output
            - `False` go further after logit to produce softmax predictions
            Note: This parameter is only useful when `include_top` is set to True.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in WEIGHTS_NAME or weights is None or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or %s' %
                         str(WEIGHTS_NAME) + ' '
                                             'or a valid path to a file containing `weights` values')

    if weights in WEIGHTS_NAME and include_top and classes != 400:
        raise ValueError('If using `weights` as one of these %s, with `include_top`'
                         ' as true, `classes` should be 400' % str(WEIGHTS_NAME))

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 4

    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(img_input, 64, 7, 7, 7, strides=(2, 2, 2), padding='same', name='Conv3d_1a_7x7')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)
    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', name='Conv3d_2b_1x1')
    x = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', name='Conv3d_2c_3x3')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x)

    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, padding='same', name='Conv3d_3b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_3b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, padding='same', name='Conv3d_3b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_3b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, padding='same', name='Conv3d_3b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, padding='same', name='Conv3d_3b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3b')

    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 192, 3, 3, 3, padding='same', name='Conv3d_3c_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_3c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 96, 3, 3, 3, padding='same', name='Conv3d_3c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_3c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3c')

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_4b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_4b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, padding='same', name='Conv3d_4b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_4b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, padding='same', name='Conv3d_4b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4b')

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4c_0a_1x1')

    branch_1 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 224, 3, 3, 3, padding='same', name='Conv3d_4c_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4c')

    # Mixed 4d
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 256, 3, 3, 3, padding='same', name='Conv3d_4d_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4d_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4d_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4d_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4d_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4d')

    # Mixed 4e
    branch_0 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4e_0a_1x1')

    branch_1 = conv3d_bn(x, 144, 1, 1, 1, padding='same', name='Conv3d_4e_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 288, 3, 3, 3, padding='same', name='Conv3d_4e_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4e_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4e_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4e_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4e_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4e')

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_4f_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4f_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_4f_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4f_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_4f_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4f_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_4f_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4f')

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x)

    # Mixed 5b
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_5b_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_5b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_5b_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_5b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5b')

    # Mixed 5c
    branch_0 = conv3d_bn(x, 384, 1, 1, 1, padding='same', name='Conv3d_5c_0a_1x1')

    branch_1 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_5c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 384, 3, 3, 3, padding='same', name='Conv3d_5c_1b_3x3')

    branch_2 = conv3d_bn(x, 48, 1, 1, 1, padding='same', name='Conv3d_5c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5c')

    if include_top:
        # Classification block
        x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
        x = Dropout(dropout_prob)(x)

        x = conv3d_bn(x, classes, 1, 1, 1, padding='same',
                      use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, classes))(x)

        # logits (raw scores for each class)
        x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)

        if not endpoint_logit:
            x = Activation('softmax', name='prediction')(x)
    else:
        h = int(x.shape[2])
        w = int(x.shape[3])
        x = AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)

    inputs = img_input
    # create model
    model = Model(inputs, x, name='i3d_inception')

    # load weights
    if weights in WEIGHTS_NAME:
        if weights == WEIGHTS_NAME[0]:  # rgb_kinetics_only
            weights_url = WEIGHTS_PATH_NO_TOP['rgb_kinetics_only']
            model_name = 'i3d_inception_rgb_kinetics_only_no_top.h5'

        elif weights == WEIGHTS_NAME[1]:  # flow_kinetics_only
            weights_url = WEIGHTS_PATH_NO_TOP['flow_kinetics_only']
            model_name = 'i3d_inception_flow_kinetics_only_no_top.h5'

        elif weights == WEIGHTS_NAME[2]:  # rgb_imagenet_and_kinetics
            weights_url = WEIGHTS_PATH_NO_TOP['rgb_imagenet_and_kinetics']
            model_name = 'i3d_inception_rgb_imagenet_and_kinetics_no_top.h5'

        elif weights == WEIGHTS_NAME[3]:  # flow_imagenet_and_kinetics
            weights_url = WEIGHTS_PATH_NO_TOP['flow_imagenet_and_kinetics']
            model_name = 'i3d_inception_flow_imagenet_and_kinetics_no_top.h5'

        downloaded_weights_path = get_file(model_name, weights_url, cache_subdir='models')
        model.load_weights(downloaded_weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model


""" The following helper functions have been added 
by https://github.com/FrederikSchorr/sign-language """


def Inception_Inflated3d_Top(input_shape: tuple, classes: int, dropout_prob: float) -> Model:
    """ Returns adjusted top layers for I3D model, depending on the number of output classes
    """

    inputs = Input(shape=input_shape, name="input")
    x = Dropout(dropout_prob)(inputs)

    x = conv3d_bn(x, classes, 1, 1, 1, padding='same',
                  use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, classes))(x)

    # logits (raw scores for each class)
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
               output_shape=lambda s: (s[0], s[2]))(x)

    # final softmax
    x = Activation('softmax', name='prediction')(x)

    # create graph of new model
    keras_model = Model(inputs=inputs, outputs=x, name="i3d_top")

    return keras_model


def add_i3d_top(base_model, classes, dropout_prob) -> Model:
    """ Given an I3D model (without top layers), this function creates the top layers
    depending on the number of output classes, and returns the entire model.
    """

    top_model = Inception_Inflated3d_Top(base_model.output_shape[1:], classes, dropout_prob)

    x = base_model.output
    predictions = top_model(x)

    new_model = Model(inputs=base_model.input, outputs=predictions, name="i3d_with_top")

    return new_model


def i3d_load(s_path: str, n_frames_norm: int, tuple_image_shape: tuple, num_class: int) -> Model:
    """ Keras load_model plus input/output shape checks
    """
    print("Load trained I3D model from %s ..." % s_path)
    keras_model = load_model(s_path)

    tu_input_shape = keras_model.input_shape[1:]
    tu_output_shape = keras_model.output_shape[1:]
    print("Loaded input shape %s, output shape %s" % (str(tu_input_shape), str(tu_output_shape)))

    if tu_input_shape != ((n_frames_norm,) + tuple_image_shape):
        raise ValueError("Unexpected I3D input shape")
    if tu_output_shape != (num_class,):
        raise ValueError("Unexpected I3D output shape")

    return keras_model

#
# def assign_tuple_value(source_tuple, index, value):
#     """
#     Assign a value to tuple at the indicated position
#     :param tuple: the tuple to alter
#     :param index: position of the value to set
#     :param value: the value to set
#     :return: modified tuple
#     """
#     temp_list = list(source_tuple)
#     temp_list[index] = value
#     return tuple(temp_list)
#
#
# def TwoStream_Inception_Inflated3d(include_top=True,
#                                    weights=None,
#                                    input_tensor=None,
#                                    flow_input_shape=None,
#                                    rgb_input_shape=None,
#                                    dropout_prob=0.0,
#                                    endpoint_logit=True,
#                                    classes=400):
#     """
#     :param include_top:
#     :param weights: List of two pre_trained model weights RGB and FLOW respectively
#     :param input_tensor:
#     :param flow_input_shape:
#     :param rgb_input_shape:
#     :param dropout_prob:
#     :param endpoint_logit:
#     :param classes:
#     :return:
#     """
#
#     if not (weights in WEIGHTS_NAME or weights is None or os.path.exists(weights)):
#         raise ValueError('The `weights` argument should be either '
#                          '`None` (random initialization) or %s' %
#                          str(WEIGHTS_NAME) + ' '
#                                              'or a valid path to a file containing `weights` values')
#
#     if weights in WEIGHTS_NAME and include_top and classes != 400:
#         raise ValueError('If using `weights` as one of these %s, with `include_top`'
#                          ' as true, `classes` should be 400' % str(WEIGHTS_NAME))
#
#     # Determine flow and rgb input shapes
#     flow_input_shape = assign_tuple_value(flow_input_shape, 3, 2)
#     rgb_input_shape = assign_tuple_value(rgb_input_shape, 3, 3)
#
#     if input_tensor is None:
#         rgb_img_input = Input(shape=rgb_input_shape)
#         flow_img_input = Input(shape=flow_input_shape)
#     else:
#         if not K.is_keras_tensor(input_tensor):
#             rgb_img_input = Input(tensor=input_tensor, shape=rgb_input_shape)
#             flow_img_input = Input(tensor=input_tensor, shape=flow_input_shape)
#         else:
#             rgb_img_input = input_tensor
#             flow_img_input = input_tensor
#
#     if K.image_data_format() == 'channels_first':
#         channel_axis = 1
#     else:
#         channel_axis = 4
#
#     flow_x = i3d_structure(flow_img_input,
#                            channel_axis,
#                            include_top,
#                            dropout_prob,
#                            endpoint_logit,
#                            classes,
#                            type='flow')
#
#     rgb_x = i3d_structure(rgb_img_input,
#                           channel_axis,
#                           include_top,
#                           dropout_prob,
#                           endpoint_logit,
#                           classes,
#                           type='rgb')
#     # create model
#     FLOW_stream = Model(input=flow_img_input, output=flow_x, name='flow_stream_i3d_inception')
#     RGB_stream = Model(input=rgb_img_input, output=rgb_x, name='rgb_stream_i3d_inception')
#
#     # load weights
#     if weights in WEIGHTS_NAME:
#         if weights == WEIGHTS_NAME[0, 1]:  # rgb_kinetics_only and flow_kinetics_only
#             if include_top:
#                 rgb_weights_url = WEIGHTS_PATH['rgb_kinetics_only']
#                 rgb_model_name = 'i3d_inception_rgb_kinetics_only.h5'
#                 flow_weights_url = WEIGHTS_PATH['flow_kinetics_only']
#                 flow_model_name = 'i3d_inception_flow_kinetics_only.h5'
#             else:
#                 rgb_weights_url = WEIGHTS_PATH_NO_TOP['rgb_kinetics_only']
#                 rgb_model_name = 'i3d_inception_rgb_kinetics_only_no_top.h5'
#                 flow_weights_url = WEIGHTS_PATH_NO_TOP['flow_kinetics_only']
#                 flow_model_name = 'i3d_inception_flow_kinetics_only_no_top.h5'
#
#         elif weights == WEIGHTS_NAME[2, 3]:  # rgb_imagenet_and_kinetics and flow_imagenet_and_kinetics
#             if include_top:
#                 rgb_weights_url = WEIGHTS_PATH['rgb_imagenet_and_kinetics']
#                 rgb_model_name = 'i3d_inception_rgb_imagenet_and_kinetics.h5'
#                 flow_weights_url = WEIGHTS_PATH['flow_imagenet_and_kinetics']
#                 flow_model_name = 'i3d_inception_flow_imagenet_and_kinetics.h5'
#             else:
#                 rgb_weights_url = WEIGHTS_PATH_NO_TOP['rgb_imagenet_and_kinetics']
#                 rgb_model_name = 'i3d_inception_rgb_imagenet_and_kinetics_no_top.h5'
#                 flow_weights_url = WEIGHTS_PATH_NO_TOP['flow_imagenet_and_kinetics']
#                 flow_model_name = 'i3d_inception_flow_imagenet_and_kinetics_no_top.h5'
#
#         downloaded_rgb_weights_path = get_file(rgb_model_name, rgb_weights_url, cache_subdir='models')
#         downloaded_flow_weights_path = get_file(flow_model_name, flow_weights_url, cache_subdir='models')
#         RGB_stream.load_weights(downloaded_rgb_weights_path)
#         FLOW_stream.load_weights(downloaded_flow_weights_path)
#
#         if K.backend() == 'theano':
#             layer_utils.convert_all_kernels_in_model(RGB_stream)
#             layer_utils.convert_all_kernels_in_model(FLOW_stream)
#
#         if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
#             warnings.warn('You are using the TensorFlow backend, yet you '
#                           'are using the Theano '
#                           'image data format convention '
#                           '(`image_data_format="channels_first"`). '
#                           'For best performance, set '
#                           '`image_data_format="channels_last"` in '
#                           'your keras config '
#                           'at ~/.keras/keras.json.')
#
#         x = RGB_stream.layers[-1].output
#         x = Flatten()(x)
#
#         y = FLOW_stream.layers[-1].output
#         y = Flatten()(y)
#
#     elif weights is not None:
#         RGB_stream.load_weights(weights[0])
#         FLOW_stream.load_weights(weights[1])
#
#         x = RGB_stream.layers[-1].output
#         x = Flatten()(x)
#
#         y = FLOW_stream.layers[-1].output
#         y = Flatten()(y)
#
#     else:  # No Weights
#         x = RGB_stream.layers[-1].output
#         x = Flatten()(x)
#
#         y = FLOW_stream.layers[-1].output
#         y = Flatten()(y)
#
#     global_stream = layers.concatenate([x, y])
#     global_stream = Dense(classes, activation='softmax', name='predictions')(global_stream)
#
#     model = Model(input=[rgb_img_input, flow_img_input], output=global_stream, name='TwoStream_I3D')
#
#     return model
