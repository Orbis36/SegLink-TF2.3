import numpy as np
from cv2 import cv2
import collections


def _generate_anchors_one_layer(h_I, w_I, h_l, w_l):
    # 传参的前两个是图像大小,后面是对应特征层大小
    # 生成两个二维矩阵，横行长度由w_l定，竖行长度由h_l来定，第一个里面为[[0,0,0,0,...][1,1,1,1,...]...],
    # 第二个是[[0,1,2,..][0,1,2,...]], 分别赋值y与x

    anchor_offset = 0.5
    gamma = 1.5

    y, x = np.mgrid[0: h_l, 0:w_l]
    cy = ((y + anchor_offset) / h_l) * h_I
    cx = ((x + anchor_offset) / w_l) * w_I
    anchor_scale = gamma * 1.0 * w_I / w_l  # 这个放大的倍数实际上决定了anchor的边长大小，也即是文中提到的经验公式，得到al
    anchor_w = np.ones_like(cx) * anchor_scale
    anchor_h = np.ones_like(cx) * anchor_scale  # cx.shape == cy.shape

    anchors = np.asarray([cx, cy, anchor_w, anchor_h])  # 这里将其想象为一个立方体，
    anchors = np.transpose(anchors, (1, 2, 0))

    return anchors

def rotate_oriented_bbox_to_horizontal(center, bbox):
    """
    Step 2 of Figure 5 in seglink paper

    Rotate bbox horizontally along a `center` point
    Args:
        center: the center of rotation
        bbox: [cx, cy, w, h, theta]
    """
    assert np.shape(center) == (2,), "center must be a vector of length 2"
    assert np.shape(bbox) == (5,) or np.shape(bbox) == (4,), "bbox must be a vector of length 4 or 5"
    bbox = np.asarray(bbox.copy(), dtype=np.float32)

    cx, cy, w, h, theta = bbox;
    M = cv2.getRotationMatrix2D(center, theta, scale=1)  # 2x3

    cx, cy = np.dot(M, np.transpose([cx, cy, 1]))

    bbox[0:2] = [cx, cy]
    return bbox

def crop_horizontal_bbox_using_anchor(bbox, anchor):
    """Step 3 in Figure 5 in seglink paper
    The crop operation is operated only on the x direction.
    Args:
        bbox: a horizontal bbox with shape = (5, ) or (4, ).
    """
    assert np.shape(anchor) == (4,), "anchor must be a vector of length 4"
    assert np.shape(bbox) == (5,) or np.shape(bbox) == (4,), "bbox must be a vector of length 4 or 5"

    # xmin and xmax of the anchor
    acx, acy, aw, ah = anchor
    axmin = acx - aw / 2.0;
    axmax = acx + aw / 2.0;

    # xmin and xmax of the bbox
    cx, cy, w, h = bbox[0:4]
    xmin = cx - w / 2.0
    xmax = cx + w / 2.0

    # clip operation
    xmin = max(xmin, axmin)
    xmax = min(xmax, axmax)

    # transform xmin, xmax to cx and w
    cx = (xmin + xmax) / 2.0;
    w = xmax - xmin
    bbox = bbox.copy()
    bbox[0:4] = [cx, cy, w, h]
    return bbox

def rotate_horizontal_bbox_to_oriented(center, bbox):
    """
    Step 4 of Figure 5 in seglink paper:
        Rotate the cropped horizontal bbox back to its original direction
    Args:
        center: the center of rotation
        bbox: [cx, cy, w, h, theta]
    Return: the oriented bbox
    """
    assert np.shape(center) == (2,), "center must be a vector of length 2"
    assert np.shape(bbox) == (5,), "bbox must be a vector of length 4 or 5"
    bbox = np.asarray(bbox.copy(), dtype=np.float32)

    cx, cy, w, h, theta = bbox
    M = cv2.getRotationMatrix2D(center, -theta, scale=1)  # 2x3
    cx, cy = np.dot(M, np.transpose([cx, cy, 1]))
    bbox[0:2] = [cx, cy]
    return bbox

def cal_seg_loc_for_single_anchor(anchor, rect):
    """
    Step 2 to 4
    """
    # rotate text box along the center of anchor to horizontal direction
    center = (anchor[0], anchor[1])
    rect = rotate_oriented_bbox_to_horizontal(center, rect)

    # crop horizontal text box to anchor
    rect = crop_horizontal_bbox_using_anchor(rect, anchor)

    # rotate the box to original direction
    rect = rotate_horizontal_bbox_to_oriented(center, rect)

    return rect

def anchor_rect_height_ratio(anchor, rect):
    """calculate the height ratio between anchor and rect
    """
    rect_height = min(rect[2], rect[3])
    anchor_height = anchor[2] * 1.0
    ratio = anchor_height / rect_height
    return max(ratio, 1.0 / ratio)

def reshape_labels_by_layer(seg_labels, feat_layers, feat_shapes):
    layer_labels = {}
    idx = 0
    for layer_name in feat_layers:
        layer_shape = feat_shapes[layer_name]
        label_length = np.prod(layer_shape)  # 特征层上像素数目，因为一一对应图中点，也即anchor数目
        layer_match_result = seg_labels[idx: idx + label_length]  # 在该特征层上的所有anchor的label
        idx += label_length

        layer_match_result = np.reshape(layer_match_result, layer_shape)  # 这是一个0-1矩阵，对应相应label的正负
        layer_labels[layer_name] = layer_match_result
    return layer_labels

def get_inter_layer_neighbours(x, y):
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
            (x - 1, y), (x + 1, y), \
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

def get_cross_layer_neighbours(x, y):
    return [(2 * x, 2 * y), (2 * x + 1, 2 * y), (2 * x, 2 * y + 1), (2 * x + 1, 2 * y + 1)]

def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >= 0 and x < w and y >= 0 and y < h

def cal_link_labels(seg_labels, feat_layers, feat_shapes):
    '''
    Input:
        seg_labels: [num_anchor,]      segment label of single img
        feat_layers: list of feature layers' name
        feat_shapes: dict, key is the name of layers in feat_layers, values are size of these layers

    Output：
        link_gt：[num_link,]
        num_link = num_anchors * 8 + (num_anchors - np.prod(feat_shapes[feat_layers[0]])) * 4
    '''
    #Train_with_igonre = True

    inter_layer_link_gts = []
    cross_layer_link_gts = []
    # 将anchor的label做成一层一层的结构,layer_label字典的key为特征层名称，值为一个size与各个层一样的数组
    layer_labels = reshape_labels_by_layer(seg_labels, feat_layers, feat_shapes)

    for layer_idx, layer_name in enumerate(feat_layers):
        layer_match_result = layer_labels[layer_name]
        h, w = feat_shapes[layer_name]
        # initalize link groundtruth for the current layer
        inter_layer_link_gt = np.ones((h, w, 8), dtype=np.int32) * (-2)
        if layer_idx > 0:  # no cross-layer link for the first layer.
            cross_layer_link_gt = np.ones((h, w, 4), dtype=np.int32) * (-2)
        for x in range(w):
            for y in range(h):
                #   layer_match_result是将segment-label resize到各个特征层大小的矩阵
                #   只有大于0的才是有效bbox，-1标志的是不清楚的，也就是igonre选项,这里也添加进去
                if layer_match_result[y, x] >= -1:
                    matched_idx = layer_match_result[y, x]  #matched_idx是在这幅图片里匹配到的bbox的index
                    # inter-layer link_gt calculation
                    # calculate inter-layer link_gt using the bbox matching result of inter-layer neighbours
                    neighbours = get_inter_layer_neighbours(x, y)
                    for nidx, nxy in enumerate(neighbours):  # 获得邻居8节点的坐标
                        nx, ny = nxy
                        if is_valid_cord(nx, ny, w, h):#这个值存在，以防止边界上的点
                            n_matched_idx = layer_match_result[ny, nx]  # 对于其每一个邻居，看看其match到了那个bbox
                            if matched_idx == n_matched_idx: #匹配到的box相同，则link为正
                                inter_layer_link_gt[y, x, nidx] = n_matched_idx #nidx为第几个邻居

                    # cross layer link_gt calculation
                    if layer_idx > 0:
                        previous_layer_name = feat_layers[layer_idx - 1]
                        ph, pw = feat_shapes[previous_layer_name]
                        previous_layer_match_result = layer_labels[previous_layer_name]
                        neighbours = get_cross_layer_neighbours(x, y)
                        for nidx, nxy in enumerate(neighbours):
                            nx, ny = nxy
                            if is_valid_cord(nx, ny, pw, ph):
                                n_matched_idx = previous_layer_match_result[ny, nx]
                                if matched_idx == n_matched_idx:
                                    cross_layer_link_gt[y, x, nidx] = n_matched_idx

        inter_layer_link_gts.append(inter_layer_link_gt)

        if layer_idx > 0:
            cross_layer_link_gts.append(cross_layer_link_gt)


    inter_layer_link_gts = np.hstack([np.reshape(t, -1) for t in inter_layer_link_gts])
    cross_layer_link_gts = np.hstack([np.reshape(t, -1) for t in cross_layer_link_gts])
    link_gt = np.hstack([inter_layer_link_gts, cross_layer_link_gts])


    return link_gt

class GetSeglinkLabel(object):

    def __init__(self,Input_size):
        self.feat_layers = ["conv4_3", "conv7", "conv6_2", "conv7_2", "conv8_2", "conv9_2"]
        self.feat_shapes = {"conv4_3": (64, 64), "conv7": (32, 32), "conv6_2": (16, 16),
                   "conv7_2": (8, 8), "conv8_2": (4, 4), "conv9_2": (2, 2)}
        self.ImageSize = Input_size
        self.MAX_HEIGHT_RATIO = 1.5
        self.prior_scaling = [0.2, 0.5, 0.2, 0.5, 20.0]
        self.anchors,self.default_anchor_center_set,self.default_anchor_map = self.GetAnchor()
        self.num_anchors = self.anchors.shape[0]
        self.num_links = self.num_anchors * 8 + (self.num_anchors - np.prod(self.feat_shapes[self.feat_layers[0]])) * 4

    def GetAnchor(self):
        all_anchors = []
        for layer in self.feat_layers:
            Feat_height, Feat_width = self.feat_shapes[layer]
            anchors = _generate_anchors_one_layer(self.ImageSize, self.ImageSize, Feat_height, Feat_width)
            all_anchors.append(anchors)

        all_anchors = [np.reshape(t, (-1, t.shape[-1])) for t in all_anchors]
        all_anchors = np.vstack(all_anchors)

        default_anchor_map = collections.defaultdict(list)
        for anchor_idx, anchor in enumerate(all_anchors):
            default_anchor_map[(int(anchor[1]), int(anchor[0]))].append(anchor_idx)
            # 生成一个字典形如{(16,16):1,....}
        default_anchor_center_set = set(default_anchor_map.keys())

        return all_anchors, default_anchor_center_set,default_anchor_map

    def __call__(self, xs, ys, *args, **kwargs):

        self.lengthOfBbox = xs.shape[0]
        xs = xs * self.ImageSize
        ys = ys * self.ImageSize
        seg_labels, seg_locations = self.match_anchor_to_test_boxes(xs, ys)

        link_labels = cal_link_labels(seg_labels, self.feat_layers, self.feat_shapes)  # 传入seg的label
        seg_offsets = self.encode_seg_offsets(seg_locations)  # 根据原文公式反推偏移量

        Train_with_igonre = True

        def set_label(label, Train_with_igonre):

            if Train_with_igonre:
                neg_need_modify = np.where(label == -2)  # -2是空值
                pos_need_modify = np.where(label > -2)  # 其余均要
                label[neg_need_modify] = 0
                label[pos_need_modify] = 1
            else:
                pos_need_modify = np.where(label > -1)  # 如果不是带着Igonre一起训练
                neg_need_modify = np.where(label <= -1)  # -1也要改掉
                label[neg_need_modify] = 0
                label[pos_need_modify] = 1

        set_label(link_labels, Train_with_igonre)
        set_label(seg_labels, Train_with_igonre)

        return seg_labels, seg_offsets, link_labels

    def match_anchor_to_test_boxes(self, xs, ys):

        #print("{} bbox".format(self.lengthOfBbox))
        seg_labels = np.ones((self.num_anchors), dtype=np.int32) * -2
        seg_locations = np.zeros((self.num_anchors, 5), dtype=np.float32)

        seg_locations[:, 2] = self.anchors[:, 2]  # 宽高一致
        seg_locations[:, 3] = self.anchors[:, 3]

        rects = self.min_area_rect(xs, ys)
        rects = self.transform_cv_rect(rects)

        bbox_mask = np.ones((512, 512), dtype=np.uint8) * (-1)
        for bbox_idx in range(self.lengthOfBbox):
            bbox_points = zip(xs[bbox_idx, :], ys[bbox_idx, :])
            box_data = np.array(list(bbox_points)).reshape([1, 4, 2]).astype(np.int32)
            cv2.fillPoly(bbox_mask, box_data, bbox_idx)

        points_in_bbox_mask = np.where(bbox_mask >= 0)
        points_in_bbox_mask = set(zip(*points_in_bbox_mask))
        points_in_bbox_mask = points_in_bbox_mask.intersection(self.default_anchor_center_set)

        for point in points_in_bbox_mask:
            anchors_here = self.default_anchor_map[point][0]  # anchor标号的获取
            bbox_idx = bbox_mask[point]
            anchor = self.anchors[anchors_here, :]
            rect = rects[bbox_idx, :]
            height_ratio = anchor_rect_height_ratio(anchor, rect)
            height_matched = height_ratio <= self.MAX_HEIGHT_RATIO
            if height_matched:
                #print("bbox_idx:{} with ratio {}".format(bbox_idx, height_ratio))
                seg_labels[anchors_here] = bbox_idx#标记为正，正数即可
                seg_locations[anchors_here, :] = cal_seg_loc_for_single_anchor(anchor, rect)
        #print("\n\n")
        return seg_labels, seg_locations

    def min_area_rect(self, xs, ys):

        box = np.empty([self.lengthOfBbox, 5])
        for index in range(self.lengthOfBbox):
            points = zip(xs[index, :], ys[index, :])
            contours = np.asarray([[list(p)] for p in points], dtype=np.int32)  # 得到组合好的点坐标
            rect = cv2.minAreaRect(contours)
            cx, cy = rect[0]  # 整个最大填充矩形的中心
            w, h = rect[1]  # 宽高
            theta = rect[2]  # 倾斜角度
            box[index, :] = [cx, cy, w, h, theta]
        return box

    def transform_cv_rect(self, rects):
        """Transform the rects from opencv method minAreaRect to our rects.
        Step 1 of Figure 5 in seglink paper

        In cv2.minAreaRect, the w, h and theta values in the returned rect are not convenient to use (at least for me), so
                the Oriented (or rotated) Rectangle object in seglink algorithm is defined different from cv2.

        Rect definition in Seglink:
            1. The angle value between a side and x-axis is:
                positive: if it rotates clockwisely, with y-axis increasing downwards.
                negative: if it rotates counter-clockwisely.
                This is opposite to cv2, and it is only a personal preference.

            2. The width is the length of side taking a smaller absolute angle with the x-axis.
            3. The theta value of a rect is the signed angle value between width-side and x-axis
            4. To rotate a rect to horizontal direction, just rotate its width-side horizontally,
                 i.e., rotate it by a angle of theta using cv2 method.
                 (see the method rotate_oriented_bbox_to_horizontal for rotation detail)


        Args:
            rects: ndarray with shape = (5, ) or (N, 5).
        Return:
            transformed rects.
        """
        only_one = False  # 只有一个
        if len(np.shape(rects)) == 1:
            rects = np.expand_dims(rects, axis=0)
            only_one = True
        assert np.shape(rects)[1] == 5, 'The shape of rects must be (N, 5), but meet %s' % (str(np.shape(rects)))

        rects = np.asarray(rects, dtype=np.float32).copy()
        num_rects = np.shape(rects)[0]
        for idx in range(num_rects):
            cx, cy, w, h, theta = rects[idx, ...]
            # assert theta < 0 and theta >= -90, "invalid theta: %f"%(theta)
            if abs(theta) > 45 or (abs(theta) == 45 and w < h):
                w, h = [h, w]
                theta = 90 + theta
            rects[idx, ...] = [cx, cy, w, h, theta]
        if only_one:
            return rects[0, ...]
        return rects

    def encode_seg_offsets(self,seg_locs):
        """
        Args:
            seg_locs: a ndarray with shape = (N, 5). It contains the abolute values of segment locations of one single image
        Return:
            seg_offsets, i.e., the offsets from default boxes. It is used as the final segment location ground truth.
        """
        anchor_cx, anchor_cy, anchor_w, anchor_h = (self.anchors[:, idx] for idx in range(4))  # 这是所有的anchor-label，经过这一步处理之后
        seg_cx, seg_cy, seg_w, seg_h = (seg_locs[:, idx] for idx in range(4))

        # encoding using the formulations from Euqation (2) to (6) of seglink paper
        #    seg_cx = anchor_cx + anchor_w * offset_cx
        offset_cx = (seg_cx - anchor_cx) * 1.0 / anchor_w

        #    seg_cy = anchor_cy + anchor_w * offset_cy
        offset_cy = (seg_cy - anchor_cy) * 1.0 / anchor_h

        #    seg_w = anchor_w * e^(offset_w)
        offset_w = np.log(seg_w * 1.0 / anchor_w)
        #    seg_h = anchor_w * e^(offset_h)
        offset_h = np.log(seg_h * 1.0 / anchor_h)

        # prior scaling can be used to adjust the loss weight of loss on offset x, y, w, h, theta
        seg_offsets = np.zeros_like(seg_locs)
        seg_offsets[:, 0] = offset_cx / self.prior_scaling[0]
        seg_offsets[:, 1] = offset_cy / self.prior_scaling[1]
        seg_offsets[:, 2] = offset_w / self.prior_scaling[2]
        seg_offsets[:, 3] = offset_h / self.prior_scaling[3]
        seg_offsets[:, 4] = seg_locs[:, 4] / self.prior_scaling[4]
        return seg_offsets
