from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Utils.model as model
from Utils.GetData import get_files

def get_one_image(train):
    #输入参数：train, 训练图片的路径
    #返回参数：image, 从训练图片中随机抽取一张图片
    n  = len(train)
    ind = np.random.randint(0, n)
    print(ind)
    img_dir = train[ind] #随机选择测试的图片
    print(img_dir)

    img = Image.open(img_dir)
    plt.imshow(img)
    imag = img.resize([64, 64])
    image = np.array(imag)
    return image

def evalute_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2  #这个数字不对会导致报错：tensorflow.python.framework.errors_impl.InvalidArgumentError: Assign requires shapes of both tensors to match. lhs shape= [128,4] rhs shape= [128,2]

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 64, 64, 3])

        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[64, 64, 3])

        logs_train_dir = 'E:/GithubProject/tensorflow-CNN-tutorials/inputdata/'

        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("Reading checkpoints..")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading succuss, global_step is %s' %global_step)
            else:
                print('No checkpoint file found')
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is a dog with possibility %.6f' %prediction[:, 0])
            elif max_index == 1:
                print('This is a cat with possibility %.6f' %prediction[:, 0])
            else:
                print('It is an accident.')

if __name__=='__main__':

    train_dir = 'E:/GithubProject/tensorflow-CNN-tutorials/inputdata'
    train, train_label, val, val_label = get_files(train_dir, 0.3)
    print(val)
    img = get_one_image(val)
    evalute_one_image(img)