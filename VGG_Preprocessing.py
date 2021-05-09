import tensorflow as tf
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

class Random_Disort_Color(object):

    # 应该注意，plt.imread和PIL.Image.open读入的都是RGB顺序，而cv2.imread读入的是BGR顺序
    def ColorSpaceChange(self,image):
        #这里应该注意，tensorflow给出的这几个API均是直接对于RGB图像，不需要转化至HSV空间
        if random.randint(0, 1):
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        if random.randint(0, 1):
            image = tf.image.random_hue(image, max_delta=0.2)
        return image

    def __call__(self, image):
        image = tf.image.random_brightness(image, max_delta=32. / 255.)

        if random.randint(0, 1):
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = self.ColorSpaceChange(image)
        else:
            image = self.ColorSpaceChange(image)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

        return image

class Random_Crop_Flip(object):
    def __init__(self):
        self.sample_options = (0.1,0.3,0.7,0.9)
        self.CROP_ASPECT_RATIO_RANGE = (0.5, 2.)
        self.AREA_RANGE = [0.1, 1]
        self.MAX_ATTEMPTS = 200
        self.OUTPUT_SIZE = 512

    def Bboxes_resize(self,bbox_ref, bboxes, xs, ys):
        h_ref = bbox_ref[2] - bbox_ref[0]  # 获得剪裁后图片宽高
        w_ref = bbox_ref[3] - bbox_ref[1]
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        xs = xs - bbox_ref[1]
        ys = ys - bbox_ref[0]

        # Scale.
        s = tf.stack([h_ref, w_ref, h_ref, w_ref])
        bboxes = bboxes / s
        xs = xs / w_ref
        ys = ys / h_ref

        xs = tf.where(xs > 0, xs, 0)
        xs = tf.where(xs < 1, xs, 1)

        ys = tf.where(ys > 0, ys, 0)
        ys = tf.where(ys < 1, ys, 1)

        return bboxes, xs, ys

    def Delete_bbox_out_boundary(self,bboxes_crop, xgs_crop, ygs_crop):
        need_delete = []
        threshold = 0.25
        bbox_ref = tf.constant([0, 0, 1, 1], tf.float32)
        for index, bbox in enumerate(bboxes_crop):
            int_ymin = tf.maximum(bbox[0], bbox_ref[0])  # 0
            int_xmin = tf.maximum(bbox[1], bbox_ref[1])  # 0.53
            int_ymax = tf.minimum(bbox[2], bbox_ref[2])  # -0.28
            int_xmax = tf.minimum(bbox[3], bbox_ref[3])  # 0.60
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            # Volumes.
            inter_vol = h * w
            bboxes_vol = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            score = inter_vol / bboxes_vol
            if score < threshold :
                need_delete.append(index)

        bboxes_crop = np.delete(bboxes_crop, need_delete, axis=0)
        xgs_crop = np.delete(xgs_crop, need_delete, axis=0)
        ygs_crop = np.delete(ygs_crop, need_delete, axis=0)

        return bboxes_crop, xgs_crop, ygs_crop

    def __call__(self, image, bbox, xs, ys, *args, **kwargs):

        # min_object_covered：（必选）数组类型为 float，默认为 0.1。图像的裁剪区域必须包含所提供的任意一个边界框的至少min_object_covered
        # 的内容。该参数的值应为非负数，当为0时，裁剪区域不必与提供的任何边界框有重叠部分。这里应该注意的是这个参数是剪裁之后的内容中所有bbox
        # 均有0.1的比例在剪裁之后的图像，且这个比例是面积，不是SSD中提到的IOU；实际上这里无法使用IOU，切割框与bbox大小差距过大
        #if random.randint(0, 1):
            #cropped_image = tf.reshape(cv2.resize(image.numpy(), (self.OUTPUT_SIZE, self.OUTPUT_SIZE)),
            #                           (self.OUTPUT_SIZE, self.OUTPUT_SIZE, 3))

         #   cropped_image = tf.image.resize(image,size=(self.OUTPUT_SIZE, self.OUTPUT_SIZE))
         #   return cropped_image, bbox, xs, ys

        ops = random.choice(self.sample_options)
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            image_size=tf.shape(image),
            bounding_boxes=tf.cast(tf.expand_dims(bbox, 0), tf.float32, name=None),
            min_object_covered = ops,
            aspect_ratio_range=self.CROP_ASPECT_RATIO_RANGE,
            area_range=self.AREA_RANGE,
            max_attempts=self.MAX_ATTEMPTS,
            use_image_if_no_bounding_boxes=True)

        # 上面返回的distort_bbox维度为[1,1,4],所以这里要重新取出，这里不是真的bbox的crop，crop的是图像本身
        # crop完成后不需要再对box做0-1规范化，我们最后用的反正也不是box的数据去训练
        distort_bbox = distort_bbox[0, 0]
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image.set_shape([None, None, 3])
        bboxes_crop, xgs_crop, ygs_crop = self.Bboxes_resize(distort_bbox, bbox, xs, ys)

        bboxes_crop, xgs_crop, ygs_crop = self.Delete_bbox_out_boundary(bboxes_crop, xgs_crop, ygs_crop)

        # cv2默认为双线性插值,宽高参数要倒过来写，再次reshape以转化为tensor，和之前images里的保持一致
        cropped_image = tf.image.resize(cropped_image,size=(self.OUTPUT_SIZE, self.OUTPUT_SIZE))
        if random.randint(0, 1):
            return cropped_image, bboxes_crop, xgs_crop, ygs_crop

        crop_flip_img = tf.image.flip_left_right(cropped_image)  # 注意这里有类型错误
        bboxes_crop[:, 1] = 1 - bboxes_crop[:, 1]
        bboxes_crop[:, 3] = 1 - bboxes_crop[:, 3]
        bboxes_crop[:, [1, 3]] = bboxes_crop[:, [3, 1]]
        xgs_crop = 1 - xgs_crop

        return crop_flip_img, bboxes_crop, xgs_crop, ygs_crop

def vgg_pre(RDC_maker, RC_FL_maker, image, bbox, xs, ys):
    
    '''
    和VGG与SSD方案类似的预处理流程
    一.图像内容变换
        1.随机亮度
        2.随机对比度
        3.随机Hue/Saturation
        4.随机Channel
    二.空间几何变换
        1.RandomCrop
        2.Flip

    这里传参进来全部是对象，以免在处理每一张图像时需要反复定义函数与变量的时间开销
    Args:
        RDC_maker: Random_Disort_Color maker
        RC_FL_maker: Random_Crop and Flip maker
    '''

    def Image_Whiten(image):
        # VGG mean parameters.
        _R_MEAN = 123.
        _G_MEAN = 117.
        _B_MEAN = 104.

        means = [_R_MEAN, _G_MEAN, _B_MEAN]
        mean = tf.constant(means, dtype=image.dtype)
        image = image*255 - mean
        image = tf.where(image < 255, image, 255)
        image = tf.where(image > 0, image, 0)

        return image


    # Convert to float scaled [0, 1].
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = RDC_maker(image)
    image, bbox, xgs, ygs = RC_FL_maker(image,bbox, xs, ys)
    image = tf.where(image < 1, image, 1)
    image = tf.where(image > 0, image, 0)
    #image = Image_Whiten(image)

    return image, bbox, xgs, ygs



