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
