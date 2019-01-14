import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

def read_cifar10_bin_data(data_path, batch_size, isTrain, shuffle):
    ''''
        read the train/test img data from the path
    '''
    img_width = 32
    img_height = 32
    img_depth = 3
    label_bytes = 1
    img_bytes = img_width * img_height * img_depth

    with tf.name_scope('input'):
        if isTrain:
            filenames = [os.path.join(data_path, 'data_batch_%d.bin' %ii) for ii in np.arange(1,6)]
        else:
            filenames = [os.path.join(data_path, 'test_batch.bin')]

        filename_queue = tf.train.string_input_producer(filenames)

        reader = tf.FixedLengthRecordReader(label_bytes + img_bytes)

        key, value = reader.read(filename_queue)

        print(key,value)

        record_bytes = tf.decode_raw(value, tf.uint8)

        label = tf.slice(record_bytes, [0], [label_bytes])
        label = tf.cast(label, tf.int32)

        image_raw = tf.slice(record_bytes, [label_bytes], [img_bytes])
        image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])
        image = tf.transpose(image_raw, (1, 2, 0))  # convert from D/H/W to H/W/D
        image = tf.cast(image, tf.float32)

        image = tf.image.per_image_standardization(image)

        if shuffle:
            image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=32,
                                                      capacity=2000,
                                                      min_after_dequeue=1500)
        else:
            image_batch, label_batch = tf.train.batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=32,
                                                      capacity=2000)
        return image_batch, tf.reshape(label_batch, [batch_size])

        #n_class = 10
        #label_batch = tf.one_hot(label_batch, depth=n_class)
        #return image_batch, tf.reshape(label_batch, [batch_size, n_class])

data_dir = 'E:/GithubProject/data_source/cifar-10-batches-bin/'
BATCH_SIZE = 10
image_batch, label_batch = read_cifar10_bin_data(data_dir,
                                        batch_size=BATCH_SIZE,
                                       isTrain=True,
                                       shuffle=True)

with tf.Session() as sess:
   i = 0
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)

   try:
       while not coord.should_stop() and i<1:

           img, label = sess.run([image_batch, label_batch])

           # just test one batch
           for j in np.arange(BATCH_SIZE):
               print(label[j])
               print('label: %d' %label[j])
               plt.imshow(img[j,:,:,:])
               plt.show()
           i+=1

   except tf.errors.OutOfRangeError:
       print('done!')
   finally:
       coord.request_stop()
   coord.join(threads)