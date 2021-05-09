import numpy as np
import cv2
import tensorflow as tf

def points_to_contour(points):
    contours = [[list(p)]for p in points]
    return np.asarray(contours, dtype = np.int32)

def points_to_contours(points):
    return np.asarray([points_to_contour(points)])

def black(shape):
    if len(np.shape(shape)) >= 2:
        shape = np.shape(shape)[0:2]
    shape = [int(v) for v in shape]
    return np.zeros(shape, np.uint8)

def bboxes_jaccard(bbox, gxs, gys):
    #     assert np.shape(bbox) == (8,)
    bbox_points = np.reshape(bbox, (4, 2))
    cnts = points_to_contours(bbox_points)

    # contruct a 0-1 mask to draw contours on
    xmax = np.max(bbox_points[:, 0])
    xmax = max(xmax, np.max(gxs)) + 10
    ymax = np.max(bbox_points[:, 1])
    ymax = max(ymax, np.max(gys)) + 10
    mask = black((ymax, xmax))

    # draw bbox on the mask
    bbox_mask = mask.copy()

    cv2.drawContours(bbox_mask, cnts, contourIdx=-1, color=1, thickness=-1)
    jaccard = np.zeros((len(gxs),), dtype=np.float32)
    # draw ground truth
    for gt_idx, gt_bbox in enumerate(zip(gxs, gys)):
        gt_mask = mask.copy()
        gt_bbox = np.transpose(gt_bbox)
        #         assert gt_bbox.shape == (4, 2)
        gt_cnts = points_to_contours(gt_bbox)
        cv2.drawContours(gt_mask, gt_cnts, contourIdx=-1, color=1, thickness=-1)

        intersect = np.sum(bbox_mask * gt_mask)
        union = np.sum(bbox_mask + gt_mask >= 1)
        #         assert intersect == np.sum(bbox_mask * gt_mask)
        #         assert union == np.sum((bbox_mask + gt_mask) > 0)
        iou = intersect * 1.0 / union
        jaccard[gt_idx] = iou
    return jaccard

def bboxes_matching(bboxes_pred, gxs, gys, gignored,matching_threshold = 0.5):
    '''
    Args:
        bboxes_pred: The output of SeglinkNet, a set of 4 numbers combination of a image
        gxs,gys: GT of x and y coordinates
        gignored: '###' of ICDAR2015, point out wheather the content of box should be noticed
    Return:
        n_gbboxes: the number we need focus(not ###)
        tp_match: 记录了对于一张图像
    '''

    gignored = tf.cast(gignored, dtype = tf.bool)
    n_gbboxes = tf.math.count_nonzero(tf.logical_not(gignored))

    gmatch = tf.zeros(np.shape(gignored), dtype = tf.bool)
    grange = tf.range(np.size(gignored), dtype = tf.int32)

    n_bboxes = tf.shape(bboxes_pred)
    rshape = (n_bboxes,)

    ta_tp_bool = tf.TensorArray(tf.bool, size=n_bboxes, dynamic_size=False, infer_shape=True)
    ta_fp_bool = tf.TensorArray(tf.bool, size=n_bboxes, dynamic_size=False, infer_shape=True)

    n_ignored_det = 0
    i = 0
    while(i<np.shape(bboxes_pred)[0]):
        rbbox = bboxes_pred[i, :]
        jaccard = bboxes_jaccard(rbbox, gxs, gys)
        idxmax = tf.cast(tf.argmax(jaccard, axis=0), dtype=tf.int32)

        jcdmax = jaccard[idxmax]
        match = jcdmax > matching_threshold
        existing_match = gmatch[idxmax]
        not_ignored = tf.logical_not(gignored[idxmax])#是否忽略的取反值

        n_ignored_det = n_ignored_det + tf.cast(gignored[idxmax], tf.int32)
        # TP: match & no previous match and FP: previous match | no match.
        # If ignored: no record, i.e FP=False and TP=False.
        tp = tf.logical_and(not_ignored, tf.logical_and(match, tf.logical_not(existing_match)))
        ta_tp_bool.write(i, tp)

        fp = tf.logical_and(not_ignored, tf.logical_or(existing_match, tf.logical_not(match)))
        ta_fp_bool.write(i, fp)

        # Update grountruth match.
        mask = tf.logical_and(tf.equal(grange, idxmax), tf.logical_and(not_ignored, match))
        gmatch = tf.logical_or(gmatch, mask)

    # TensorArrays to Tensors and reshape.
    tp_match = tf.reshape(ta_tp_bool.stack(), rshape)
    fp_match = tf.reshape(ta_fp_bool.stack(), rshape)

    return n_gbboxes, tp_match, fp_match



