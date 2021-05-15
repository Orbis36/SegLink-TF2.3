import tensorflow as tf
import numpy as np
import os
import sys

def Get_dir(data_path):
    imagelist = []
    txtlist = []
    for (root, dirs, files) in os.walk(data_path):
        for name in files:
            if name.lower().endswith(('.jpg')):
                name = os.path.join(root, name)
                imagelist.append(name)

            if name.lower().endswith(('.txt')):
                name = os.path.join(root, name)
                txtlist.append(name)
    # 按数字顺序排列
    imagelist.sort(key=lambda x: int((x.split('.')[0]).split('_')[-1]))
    txtlist.sort(key=lambda x: int((x.split('.')[0]).split('_')[-1]))

    return imagelist, txtlist

def Getlabel(txt_dir,Mode = 'ICDAR2015',ImgSize = None):
    bboxes = []
    xgs = []
    ygs = []

    txt_file = open(txt_dir, "r", encoding='utf-8-sig')
    for line in txt_file.readlines():
        if '\xef\xbb\xbf' in line:
            line.replace('\xef\xbb\xbf', '')

        if Mode == 'ICDAR2015':
            gt = line.split(',')
            oriented_box = [int(gt[i]) for i in range(8)]
            oriented_box = np.asarray(oriented_box) / ([1280, 720] * 4)  # 为了预处理中满足tf的要求,这个数据集是固定大小的图像
        if Mode == 'ICDAR2013':
            line = line.replace(',','')
            gt = line.split(' ')
            oriented_box = [int(gt[0]),int(gt[1]),int(gt[2]),int(gt[1]),int(gt[2]),int(gt[3]),int(gt[0]),int(gt[3])]
            oriented_box = np.asarray(oriented_box) / ([ImgSize[1], ImgSize[0]] * 4)  # 为了预处理中满足tf的要求,这个数据集是固定大小的图像

        xs = np.array(oriented_box).reshape(4, 2)[:, 0]  # 这里的xs与ys的保存时有必要的，需要按照相关的大小计算anchor
        ys = np.array(oriented_box).reshape(4, 2)[:, 1]
        xmin = xs.min()
        xmax = xs.max()
        ymin = ys.min()
        ymax = ys.max()
        bbox = np.array([ymin, xmin, ymax, xmax])
        # focus = '###' not in gt[-1]#不使用
        bboxes.append(bbox)
        xgs.append(xs)
        ygs.append(ys)

    return np.array(bboxes), np.array(xgs), np.array(ygs)

if __name__ == "__main__":

    # dir of train data should be the first
    if not os.path.exists('./DatasetInTFRecord/ICDAR2015'):
        os.mkdir('./DatasetInTFRecord/ICDAR2015')

    os.chdir('./DatasetInTFRecord/ICDAR2015')
    work_dir = os.getcwd()
    TFRecord_dir = ["ICDAR2015_train.tfrecords", "ICDAR2015_test.tfrecords"]
    Data_dir = [sys.argv[1],sys.argv[2]]

    for tf_dir,data_dir in zip(TFRecord_dir,Data_dir):
        with tf.io.TFRecordWriter(os.path.join(work_dir,tf_dir)) as writer:
            imgs_name, labels_name = Get_dir(data_dir)
            for img_name, label_name in zip(imgs_name, labels_name):
                bbox_pre, xs, ys = Getlabel(label_name)
                img = tf.io.read_file(img_name).numpy()
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








