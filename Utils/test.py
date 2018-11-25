from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Utils.model
from Utils.GetData import get_files

def get_one_image(train):
    #输入参数：train, 训练图片的路径
    #返回参数：image, 从训练图片中随机抽取一张图片
    n  = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind] #随机选择测试的图片

    img = Image.open(img_dir)
    plt.imshow(img)
    imag = img.resize([64, 64])
    image = np.array(imag)
    return image

def evalute_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 4

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 64, 64, 3])
        