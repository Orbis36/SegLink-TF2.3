import tensorflow as tf
import numpy as np
import SeglinkNet
import matplotlib.pyplot as plt
import cv2

from Seglink import GetSeglinkLabel
from SeglinkToBbox import seglink2bbox
from VGG_Preprocessing import Random_Crop_Flip,Random_Disort_Color,vgg_pre

tfrecord_file = 'C:\Code\GraduateThesis\SegLink - Final\CreateGT\DatasetInTFRecord\ICDAR2015\ICDAR2015_train.tfrecords'
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

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

    return image,bboxes, xs, ys

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

RDC_maker = Random_Disort_Color()
RCF_maker = Random_Crop_Flip()
SL_maker = GetSeglinkLabel(512)
dataset = raw_dataset.map(_parse_example)
for image, bboxes, xs, ys in dataset.take(5):

    image, bbox_vgg, xgs, ygs = vgg_pre(RDC_maker, RCF_maker, image, bboxes, xs, ys)
    break

seg_labels, seg_offsets, link_labels = SL_maker(xgs, ygs)
label = np.where(seg_labels>0)[0]
print(label)
ShowBBox(image, bbox_vgg)

image = tf.expand_dims(image, 0)
SLNetModel = SeglinkNet.SegLinkNet(Pretrain = './SegLink_iteration140000.h5')
SLNetModel((512,512))#build

Endpoint = SLNetModel.GetEndPoint(image)
seg_score_logits, seg_offsets, seg_scores, link_scores, link_score_logits = SLNetModel._add_seglink_estimator(Endpoint)

seg_scores = seg_scores.numpy()
link_scores = link_scores.numpy()
seg_offsets = seg_offsets.numpy()
image = image[0].numpy()

print(np.where(seg_scores[0][:,1]>0.5))
#print(seg_offsets.shape)
bboxes = seglink2bbox(seg_scores[0][:,1], link_scores[0][:,1], seg_offsets[0], (512,512), 0.5,0.5)

for bbox in bboxes:
    bbox = np.reshape(bbox, (4, 2))
    cnts = np.asarray([np.asarray([[list(p)] for p in bbox], dtype=np.int32)])
    image = cv2.drawContours(image, cnts, contourIdx=-1, color=(0, 255, 0), thickness=1)

cv2.imshow('imageshow', image)  # 显示返回值image，其实与输入参数的thresh原图没啥区别
cv2.waitKey(0)