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

#池化层1
#3x3最大池化，步长strides为2，池化后执行lrn()操作，局部响应归一化，对训练有利。<lrn()是什么？>
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

#卷积层2
#16个3x3的卷积核（16通道），padding='SAME'，激活函数relu
    with tf.variable_scope('conv2') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3,3,64,16], stddev=0.1, dtype=tf.float32),
                              name='weights', dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[16]),
                             name='biases', dtype=tf.float32)
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

#池化层2
#3x3 最大池化
    with tf.variable_scope('pool2_lrn') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pooling2')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')

#全连接层3
#128个神经元，将之前pool层的输出reshape成一行，激活函数relu()
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal(shape=[dim,128], stddev=0.005, dtype=tf.float32),
                              name='weights', dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                             name='biases', dtype=tf.float32)
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope._name)

#全连接层4
#128个神经元，激活函数relu()
    with tf.variable_scope('local4') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[128,128], stddev=0.005, dtype=tf.float32),
                              name='weights', dtype=tf.float32)
        biases = tf.Variable(tf.constant(value = 0.1, dtype = tf.float32, shape=[128]),
                             name='biases', dtype=tf.float32)
        local4 = tf.nn.relu(tf.matmul(local3, weights)+biases, name='local4')

#dropout层
#   with tf.variable_scope('dropout') as scope:
#       drop_out = tf.nn.dropout(local4, 0.8)

#Softmax回归层
#将前面的FC层输出，做一个线性回归，计算出每一类的得分，在这里是两类，所以这个层输出的是两个得分。
