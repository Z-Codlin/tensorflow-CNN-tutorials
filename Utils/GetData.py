import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_dir = 'E:/GithubProject/tensorflow-CNN-tutorials/inputdata/'

dog = []
label_dog = []
cat = []
label_cat = []

#step1: 获取‘input_data/’下面所有的图片路径名，存放到对应地列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir, radio):
    for file in os.listdir(file_dir+'/dog/'):
        dog.append(file_dir+'/dog'+'/'+file)
        label_dog.append(0)
    for file in os.listdir(file_dir+'/cat/'):
        cat.append(file_dir+'/cat'+'/'+file)
        label_cat.append(1)
    #step2: 对生成的图片路径和标签List做打乱处理，把cat和dog合起来组成一个list(img和label)
    image_list = np.hstack((dog, cat))
    label_list = np.hstack((label_dog, label_cat))

    #利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    #将所有的img和label转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    #将所得List分为两部分，一部分用来训练，一部分用来测试
    #radio是测试集的比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * radio))
    n_train = n_sample - n_val

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_lables = all_label_list[n_train:-1]
    val_lables = [int(float(i)) for i in val_lables]

    return tra_images, tra_labels, val_images, val_lables

#-----------------生成batch------------------

#step1. 将上面生成的list传入get_batch(),转换类型，产生一个输入队列queue.因为img和label是分开的，
#·所以使用tf.train.slice_input_producer(),然后用tf.read_file()从队列中读取图像。
#·image_W, image_H :设置好固定的图像高度和宽度。
#   设置batch_size:每个batch要放多少张图片
#   capacity: 一个队列最大多少
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue


# step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_and_crop_jpeg(image_contents, channels=3)

# step3: 数据预处理，对图像进行旋转，缩放，裁剪，归一化等操作，让计算出的模型更健壮
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

#step4：生成batch
#image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
#label_batch: 1D tensor [batch_size], dtype=tf.int32

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size= batch_size,
                                              num_threads= 32,
                                              capacity = capacity)
    #重新排列label, 行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch
