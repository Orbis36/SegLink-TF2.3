import tensorflow as tf


feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'image': tf.io.FixedLenFeature([], tf.string),
    'bbox_num': tf.io.FixedLenFeature([], tf.int64),
    'xs': tf.io.VarLenFeature(tf.float32),
    'ys': tf.io.VarLenFeature(tf.float32),
    'bbox_pre': tf.io.VarLenFeature(tf.float32)
}

def _parse_example(example_string): # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    feature_dict = tf.io.parse_single_example(example_string, feature_description)

    objects_number = tf.cast(feature_dict['bbox_num'], tf.int32)
    bboxes_shape = tf.parallel_stack([objects_number, 4])

    bboxes = tf.sparse.to_dense(feature_dict['bbox_pre'], default_value=0)
    xs = tf.sparse.to_dense(feature_dict['xs'], default_value=0)
    ys = tf.sparse.to_dense(feature_dict['ys'], default_value=0)

    bboxes = tf.reshape(bboxes, bboxes_shape)
    xs = tf.reshape(xs, bboxes_shape)
    ys = tf.reshape(ys, bboxes_shape)
    image = tf.io.decode_jpeg(feature_dict['image'])  # 解码JPEG图片

    return image, bboxes, xs, ys

if __name__ == "__main__":

    dataset = tf.data.TFRecordDataset('./CreateGT/DatasetInTFRecord/ICDAR2015/ICDAR2015_train.tfrecords')
    dataset = dataset.map(_parse_example).shuffle(1000,reshuffle_each_iteration=True).\
        prefetch(tf.data.experimental.AUTOTUNE)
    count = 0
    for image, bboxes, xs, ys in dataset.take(2000):
        count+=1
    print(count)