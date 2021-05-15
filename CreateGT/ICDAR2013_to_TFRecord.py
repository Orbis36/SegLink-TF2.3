import tensorflow as tf
import os
import sys
import numpy as np
from ICDAR2015_to_TFRecord import Get_dir,Getlabel
import matplotlib.pyplot as plt

def ShowBBox(img_data1, bboxes):

    colors = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    batched = tf.expand_dims(
        tf.image.convert_image_dtype(img_data1, tf.float32), 0
    )
    bboxes = np.array(bboxes).reshape(1, -1, 4)

    result = tf.image.draw_bounding_boxes(images=batched, colors=colors, boxes=bboxes)
    result = tf.reduce_sum(result, 0)
    plt.rcParams['figure.dpi'] = 200
    plt.imshow(result.numpy())
    plt.show()

if __name__ == "__main__":
    if not os.path.exists('./DatasetInTFRecord/ICDAR2013'):
        os.mkdir('./DatasetInTFRecord/ICDAR2013')
    os.chdir('./DatasetInTFRecord/ICDAR2013')
    work_dir = os.getcwd()
    TFRecord_dir = ["ICDAR2013_train.tfrecords", "ICDAR2013_test.tfrecords"]
    Data_dir = [sys.argv[1], sys.argv[2]]

    for tf_dir, data_dir in zip(TFRecord_dir, Data_dir):
        with tf.io.TFRecordWriter(os.path.join(work_dir, tf_dir)) as writer:
            imgs_name, labels_name = Get_dir(data_dir)
            for img_name, label_name in zip(imgs_name, labels_name):
                img_raw = tf.io.read_file(img_name)
                img = img_raw.numpy()
                shape = tf.io.decode_jpeg(img_raw).shape[0:2]
                bbox_pre, xs, ys = Getlabel(label_name,Mode='ICDAR2013',ImgSize=shape)
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                    'bbox_num': tf.train.Feature(int64_list=tf.train.Int64List(value=[bbox_pre.shape[0]])),
                    'xs': tf.train.Feature(float_list=tf.train.FloatList(value=list(xs.reshape(-1)))),
                    'ys': tf.train.Feature(float_list=tf.train.FloatList(value=list(ys.reshape(-1)))),
                    'bbox_pre': tf.train.Feature(float_list=tf.train.FloatList(value=list(bbox_pre.reshape(-1))))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立 Example
                writer.write(example.SerializeToString())
    print('done')