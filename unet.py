import numpy as np
import tensorflow as tf


def conv3x3(input, filters):
    return tf.nn.relu(tf.nn.conv2d(input, filters, 1, "VALID", "NHWC"))


def maxpool(input):
    return tf.nn.max_pool(input, 2, 2, "SAME", "NHWC")


def upconv(input, filters):
    in_shape = input.shape()
    out_shape = tf.constant([in_shape[1]*2, in_shape[2]*2])
    return tf.nn.conv2d(tf.image.resize(input, out_shape, method=ResizeMethod.NEAREST), filters, 1, "VALID", "NHWC")


def conv1x1(input, filters):
    return tf.nn.conv2d(input, filters, 1, "VALID", "NHWC")


def copycrop(l, r):
    r_shape = r.shape()
    rh = r_shape[1]
    rw = r_shape[2]
    l_shape = l.shape)
    lh = l_shape[1]
    rh = r_shape[2]
    l_crop = tf.image.crop_to_bounding_box(l, ceil((lh - rh) / 2), ceil((lw - rw) / 2), rh, rw)
    return tf.concat([l_crop, r], 3)


num_classes = 1
initial_weight_mean = 0
initial_weight_stddev = 1
in_layers = 2

stage1_layers = 64
stage2_layers = stage1_layers
stage3_layers = stage2_layers * 2
stage4_layers = stage3_layers
stage5_layers = stage4_layers * 2
stage6_layers = stage5_layers
stage7_layers = stage6_layers * 2
stage8_layers = stage7_layers
stage9_layers = stage8_layers * 2
stage10_layers = stage9_layers
stage11_layers = stage10_layers / 2
stage12_layers = stage11_layers * 2
stage13_layers = stage13_layers / 2
stage14_layers = stage13_layers
stage15_layers = stage14_layers / 2
stage16_layers = stage15_layers * 2
stage17_layers = stage16_layers / 2
stage18_layers = stage17_layers
stage19_layers = stage18_layers / 2
stage20_layers = stage19_layers * 2
stage21_layers = stage20_layers / 2
stage22_layers = stage21_layers
stage23_layers = stage22_layers / 2
stage24_layers = stage23_layers * 2
stage25_layers = stage24_layers / 2
stage26_layers = stage25_layers
stage27_layers = num_classes

def conv3x3_filter_shape(in_layers, out_layers):
    return [3, 3, in_layers, out_layers]

def conv2x2_filter_shape(in_layers, out_layers):
    return [2, 2, in_layers, out_layers]

def conv1x1_filter_shape(in_layers, out_layers);
    return [1, 1, in_layers, out_layers]

filters_shape = [ conv3x3_filter_shape(in_layers, stage1_layers)
                , conv3x3_filter_shape(stage1_layers, stage2_layers)
                , conv3x3_filter_shape(stage2_layers, stage3_layers)
                , conv3x3_filter_shape(stage3_layers, stage4_layers)
                , conv3x3_filter_shape(stage4_layers, stage5_layers)
                , conv3x3_filter_shape(stage5_layers, stage6_layers)
                , conv3x3_filter_shape(stage6_layers, stage7_layers)
                , conv3x3_filter_shape(stage7_layers, stage8_layers)
                , conv3x3_filter_shape(stage8_layers, stage9_layers)
                , conv3x3_filter_shape(stage9_layers, stage10_layers)
                , conv2x2_filter_shape(stage10_layers, stage11_layers)
                , conv3x3_filter_shape(stage12_layers, stage13_layers)
                , conv3x3_filter_shape(stage13_layers, stage14_layers)
                , conv2x2_filter_shape(stage14_layers, stage15_layers)
                , conv3x3_filter_shape(stage16_layers, stage17_layers)
                , conv3x3_filter_shape(stage17_layers, stage18_layers)
                , conv2x2_filter_shape(stage18_layers, stage19_layers)
                , conv3x3_filter_shape(stage20_layers, stage21_layers)
                , conv3x3_filter_shape(stage21_layers, stage22_layers)
                , conv2x2_filter_shape(stage22_layers, stage23_layers)
                , conv3x3_filter_shape(stage24_layers, stage25_layers)
                , conv3x3_filter_shape(stage25_layers, stage26_layers)
                , conv1x1_filter_shape(stage26_layers, stage27_layers) ]


class UNet:
    # input is a tensor of shape [batch_size, height, width, in_layers]
    # params is a 1D tensor
    def run(input, params):
        filters = tf.RaggedTensor.from_nested_row_lengths(params, filters_shape)
        stage1_filter = filters[0]
        stage2_filter = filters[1]
        stage3_filter = filters[2]
        stage4_filter = filters[3]
        stage5_filter = filters[4]
        stage6_filter = filters[5]
        stage7_filter = filters[6]
        stage8_filter = filters[7]
        stage9_filter = filters[8]
        stage10_filter = filters[9]
        stage11_filter = filters[10]
        stage13_filter = filters[11]
        stage14_filter = filters[12]
        stage15_filter = filters[13]
        stage17_filter = filters[14]
        stage18_filter = filters[15]
        stage19_filter = filters[16]
        stage21_filter = filters[17]
        stage22_filter = filters[18]
        stage23_filter = filters[19]
        stage25_filter = filters[20]
        stage26_filter = filters[21]
        stage27_filter = filters[22]
        stage1 = conv3x3(input, stage1_filter)
        stage2a = conv3x3(stage1, stage2_filter)
        stage2b = maxpool(stage2a)
        stage3 = conv3x3(stage2b, stage3_filter)
        stage4a = conv3x3(stage3, stage4_filter)
        stage4b = maxpool(stage4a)
        stage5 = conv3x3(stage4b, stage5_filter)
        stage6a = conv3x3(stage5, stage6_filter)
        stage6b = maxpool(stage6a)
        stage7 = conv3x3(stage6b, stage7_filter)
        stage8a = conv3x3(stage7, stage8_filter)
        stage8b = maxpool(stage8a)
        stage9 = conv3x3(stage8b, stage9_filter)
        stage10 = conv3x3(stage9, stage10_filter)
        stage11 = upconv(stage10, stage11_filter)
        stage12 = copycrop(stage8a, stage11)
        stage13 = conv3x3(stage12, stage13_filter)
        stage14 = conv3x3(stage13, stage14_filter)
        stage15 = upconv(stage14, stage15_filter)
        stage16 = copycrop(stage6a, stage15)
        stage17 = conv3x3(stage16, stage17_filter)
        stage18 = conv3x3(stage17, stage18_filter)
        stage19 = upconv(stage18, stage19_filter)
        stage20 = copycrop(stage4a, stage19)
        stage21 = conv3x3(stage20, stage21_filter)
        stage22 = conv3x3(stage21, stage22_filter)
        stage23 = upconv(stage22, stage23_filter)
        stage24 = copycrop(stage2a, stage23)
        stage25 = conv3x3(stage24, stage25_filter)
        stage26 = conv3x3(stage25, stage26_filter)
        stage27 = conv1x1(stage26, stage27_filter)
        return stage27
