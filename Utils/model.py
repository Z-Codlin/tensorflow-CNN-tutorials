#-*-endcoding:utf-8-*-

# 输入数据：(batch_size, IMG_W, IMG_H, col_channel) = (20, 64, 64, 3)
# 卷积层1：(conv_kernel, num_channel, num_out_neure) = (3, 3, 3, 64)
# 池化层1：(ksize, strides, padding) = ([1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
# 卷积层2：(conv_kernel, num_channel, num_out_neure) = (3, 3, 64, 16)
# 池化层2：(ksize, strides, padding) = ([1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
# 全连接1：(out_pool2_reshape, num_out_neure) = (dim, 128)
# 全连接2：(fc1_out, num_out_nuere) = (128, 128)
# softmax层：(fc2_out, num_classes) = (128, 2)  #我只用两类：猫和狗
#
# 激活函数：tf.nn.relu
# 损失函数：tf.nn.sparse_softmax_cross_entropy_with_logits

import tensorflow as tf

#网络结构定义：
    #输入参数：images, image batch, 4D tensor, tf.float32,
    #返回参数：logits, float, [batch_size, n_classes]
def inference(images, batch_size, n_classes):
    #一个简单的卷积神经网络，卷积+池化层x2,全连接层x2, 最后一个softmax层做分类。
    #卷积层1
    #64个3x3的卷积核（3通道），padding='SAME',表示padding后卷积的图与原尺寸一致，激活函数relu
    with tf.variable_scope('conv1') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3,3,3,64], stddev= 1.0, dtype= tf.float32),
                              name = 'weights', dtype= tf.float32)
        biases = tf.Variable(tf.constant(value = 0.1, dtype = tf.float32, shape = [64]),
                             name = 'biases', dtype = tf.float32)
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)

    with tf.variable_scope('pooling1_lrn') as scope: