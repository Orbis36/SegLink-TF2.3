import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import os
import random


class SynthText2TFRecord():
    def __init__(self, mat_path, root_path):
        self.mat_path = mat_path
        self.root_path = root_path
        self._load_mat()

    def _load_mat(self):
        data = loadmat(self.mat_path)
        combo = list(zip(data['imnames'][0], data['wordBB'][0], data['txt'][0]))
        random.shuffle(combo)
        self.image_paths, self.image_bbox, self.txts = zip(*combo)
        self.num_images = len(self.image_paths)

    def get_record(self, image_idx):
        image_path = os.path.join(self.root_path, self.image_paths[image_idx][0])
        if not os.path.exists(image_path):
            return None
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        h, w = img.shape[0:-1]
        num_words = self.get_num_words(image_idx)
        rect_bboxes = []
        xs = []
        ys = []
        for word_idx in range(num_words):
            xys = self.get_word_bbox(image_idx, word_idx)
            is_valid, min_x, min_y, max_x, max_y, xys = self.normalize_bbox(xys, width=w, height=h)
            if not is_valid:
                continue
            rect_bboxes.append([min_y, min_x, max_y, max_x])

            xys = np.reshape(np.transpose(xys), -1)
            xs.append(xys[::2])
            ys.append(xys[1:][::2])

        if len(rect_bboxes) == 0:
            return None
        return image_path, np.asarray(rect_bboxes), np.asarray(xs), np.asarray(ys)

    def get_word_bbox(self, img_idx, word_idx):
        boxes = self.image_bbox[img_idx]
        if len(np.shape(boxes)) == 2:  # error caused by dataset
            boxes = np.reshape(boxes, (2, 4, 1))

        xys = boxes[:, :, word_idx]
        assert (np.shape(xys) == (2, 4))
        return np.float32(xys)

    def get_num_words(self, idx):
        try:
            return np.shape(self.image_bbox[idx])[2]
        except:  # error caused by dataset
            return 1

    def normalize_bbox(self, xys, width, height):
        xs = xys[0, :]
        ys = xys[1, :]

        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)

        # bound them in the valid range
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(width, max_x)
        max_y = min(height, max_y)

        # check the w, h and area of the rect
        w = max_x - min_x
        h = max_y - min_y
        is_valid = True

        if w < 10 or h < 10:
            is_valid = False

        if w * h < 100:
            is_valid = False

        xys[0, :] = xys[0, :] / width
        xys[1, :] = xys[1, :] / height

        return is_valid, min_x / width, min_y / height, max_x / width, max_y / height, xys


if __name__ == "__main__":
    mat_path = 'C:\Code\GraduateThesis\Dataset-origin\SynthText\SynthText\data\SynthText\gt.mat'
    root_path = 'C:\Code\GraduateThesis\Dataset-origin\SynthText\SynthText\data\SynthText'
    output_dir = './DatasetInTFRecord/SynthText'

    records_per_file = 50000
    image_idx = -1
    tfrecord_id = 0 #需要切割成多个tfrecord
    tf_maker = SynthText2TFRecord(mat_path, root_path)

    while image_idx < tf_maker.num_images:
        with tf.io.TFRecordWriter(output_dir+'/SynthText_{}.tfrecords'.format(tfrecord_id)) as writer:
            record_count = 0
            while record_count != records_per_file:
                image_idx += 1
                if image_idx >= tf_maker.num_images:
                    break
                #找到img_idx的记录
                record = tf_maker.get_record(image_idx)
                if record is None:
                    print('\nimage {} does not exist'.format(image_idx + 1))
                    continue

                image_path, rect_bboxes, xs, ys = record
                img = tf.io.read_file(image_path).numpy()
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                    'bbox_num': tf.train.Feature(int64_list=tf.train.Int64List(value=[rect_bboxes.shape[0]])),
                    'bbox_pre': tf.train.Feature(float_list=tf.train.FloatList(value=list(rect_bboxes.reshape(-1)))),
                    'xs': tf.train.Feature(float_list=tf.train.FloatList(value=list(xs.reshape(-1)))),
                    'ys': tf.train.Feature(float_list=tf.train.FloatList(value=list(ys.reshape(-1)))),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立 Example
                writer.write(example.SerializeToString())
                record_count += 1

                if image_idx%1000 == 0:
                    print("{}/{} 完成".format(image_idx,tf_maker.num_images))
        tfrecord_id += 1



