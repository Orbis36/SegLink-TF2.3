import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os

import SeglinkNet
from VGG_Preprocessing import Random_Disort_Color,Random_Crop_Flip,vgg_pre
from Seglink import GetSeglinkLabel



physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'image': tf.io.FixedLenFeature([], tf.string),
    'bbox_num': tf.io.FixedLenFeature([], tf.int64),
    'xs': tf.io.VarLenFeature(tf.float32),
    'ys': tf.io.VarLenFeature(tf.float32),
    'bbox_pre': tf.io.VarLenFeature(tf.float32)
}
RDC_maker = Random_Disort_Color()
RCF_maker = Random_Crop_Flip()
SL_maker = GetSeglinkLabel(512)

optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=1e-4)
@tf.function
def train_step(SLNetModel,b_IMAGE,b_Seg_Label,b_Seg_offests,b_Link_label):

    with tf.GradientTape() as tape:
        Endpoint = SLNetModel.GetEndPoint(b_IMAGE)
        seg_score_logits, seg_offsets, seg_scores, link_scores, link_score_logits = SLNetModel._add_seglink_estimator(Endpoint)
        seg_cls_loss, seg_loc_loss, link_cls_loss = \
            SLNetModel.build_loss(b_Seg_Label,b_Seg_offests,b_Link_label,seg_score_logits, seg_offsets, seg_scores, link_scores, link_score_logits)
        Loss = seg_cls_loss + seg_loc_loss + link_cls_loss


    gradients = tape.gradient(Loss, SLNetModel.trainable_variables)
    optimizer.apply_gradients(zip(gradients, SLNetModel.trainable_variables))
    return seg_cls_loss,seg_loc_loss,link_cls_loss

def Preprocessing(image, bboxes, xs, ys):
    image, bbox, xgs, ygs = vgg_pre(RDC_maker, RCF_maker, image, bboxes, xs, ys)
    seg_labels, seg_offsets, link_labels = SL_maker(xgs, ygs)
    return image, seg_labels, seg_offsets, link_labels

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

def ShowBBox(img_data1, bboxes):

    colors = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    img_data1 = tf.expand_dims(tf.image.convert_image_dtype(img_data1[0], tf.float32), 0)
    result = tf.image.draw_bounding_boxes(images=img_data1, colors=colors, boxes=bboxes)
    result = tf.reduce_sum(result, 0)
    plt.rcParams['figure.dpi'] = 200
    plt.imshow(result.numpy())
    plt.show()


if __name__ == "__main__":

    '''
    SynthText已经在转TFRecord时打乱顺序
    这里dataset载入只用一次，先载入数据，每1000个取一个再做一次shuffle
    所以这里repeat就表示在多
    
    这里问题是我要对于每一个batch做预处理而非一次性做完
    而且对内存的要求使得这里不能保留raw_dataset
    '''

    BATCH_SIZE = 4
    iteration = 0
    count = False
    ini_time = datetime.datetime.now()
    filePath = './CreateGT/DatasetInTFRecord/SynthText/'
    log_dir = "./Seglink" + ini_time.strftime("%Y%m%d-%H%M%S")
    PreTrainDir = './SegLink_iteration85000.h5'

    file_list = os.listdir(filePath)
    abs_file_list = [os.path.abspath(filePath + x) for x in file_list]
    summary_writer = tf.summary.create_file_writer(log_dir)  # 创建日志文件句柄

    SLNetModel = SeglinkNet.SegLinkNet(PreTrainDir)
    SLNetModel((512,512))#build
    Temp_Dataset = tf.queue.FIFOQueue(capacity=BATCH_SIZE,dtypes=[tf.float32, tf.int32, tf.int32, tf.float32],
                                      shapes=[(512,512,3),(5460,),(49136,),(5460, 5)])
    dataset = tf.data.TFRecordDataset(abs_file_list)
    dataset = dataset.map(_parse_example).shuffle(1000,reshuffle_each_iteration=True).repeat(2).\
        prefetch(tf.data.experimental.AUTOTUNE)


    for b_IMAGE, b_Bboxes, b_Xs, b_Ys in dataset:   #取出一个batch
        image, seg_labels, seg_offsets, link_labels = Preprocessing(b_IMAGE, b_Bboxes, b_Xs, b_Ys)
        del b_IMAGE, b_Bboxes, b_Xs, b_Ys
        Temp_Dataset.enqueue([image, seg_labels, link_labels, seg_offsets])
        if Temp_Dataset.size() == BATCH_SIZE:
            b_IMAGE_use, b_Seg_Label_use, b_Link_label_use, b_Seg_offests_use = Temp_Dataset.dequeue_many(BATCH_SIZE)
            seg_cls_loss, seg_loc_loss, link_cls_loss = train_step(SLNetModel, b_IMAGE_use, b_Seg_Label_use,
                                                                   b_Seg_offests_use, b_Link_label_use)
            loss = seg_cls_loss + seg_loc_loss + link_cls_loss
            with summary_writer.as_default():  # 将loss写入TensorBoard
                tf.summary.scalar('seg_cls_loss', seg_cls_loss, step=iteration)
                tf.summary.scalar('seg_loc_loss', seg_loc_loss, step=iteration)
                tf.summary.scalar('link_cls_loss', link_cls_loss, step=iteration)
                tf.summary.scalar('total_loss', loss, step=iteration)
            if iteration % 10 == 0:
                print("iteration {} Loss {}".format(iteration, loss))
                if iteration > 60000 and (iteration % 5000 == 0):
                    SLNetModel.save_weights('./SegLink_iteration{}.h5'.format(iteration), save_format='h5')
                if iteration >= 90000:
                    SLNetModel.save_weights('./SegLink_iteration{}.h5'.format(iteration), save_format='h5')
                    break
    end_time = datetime.datetime.now()
    print("End of training, totally use: {} minutes".format((end_time - ini_time).seconds / 60))

