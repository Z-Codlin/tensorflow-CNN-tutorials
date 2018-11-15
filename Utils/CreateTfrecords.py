import os
import tensorflow as tf
from PIL import Image

origin_picture = 'E:/GithubProject/tensorflow-CNN-tutorials/DataSource/'

gen_picture = 'E:/GithubProject/tensorflow-CNN-tutorials/inputdata/'

classes = {'dog', 'cat'}

num_samples = 200

#制作TFRecords数据
def create_record():
    writer = tf.python_io.TFRecordWriter("cat_dog_classify.tfrecords")
    for index, name in enumerate(classes):
        class_path = origin_picture+"/"+name+"/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((64, 64))
            img_raw = img.tobytes() #将图片转化为原生bytes
            print(index, img_raw)
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [64, 64, 3])

    label = tf.cast(label, tf.int32)
    return img, label

if __name__ == '__main__':
    create_record()
    batch = read_and_decode('cat_dog_classify.tfrecords')
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num_samples):
            example, lab = sess.run(batch)
            img = Image.fromarray(example, 'RGB')
            img.save(gen_picture+'/'+str(i)+'sample'+str(lab)+'.jpg')
            print(example, lab)
        coord.request_stop()
        coord.join(threads)
        sess.close()