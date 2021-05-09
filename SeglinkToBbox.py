import cv2
import numpy as np
from Seglink import reshape_labels_by_layer,get_inter_layer_neighbours,is_valid_cord,\
    get_cross_layer_neighbours,_generate_anchors_one_layer

def reshape_link_gt_by_layer(link_gt,feat_layers,feat_shapes):
    inter_layer_link_gts = {}
    cross_layer_link_gts = {}

    idx = 0
    for layer_idx, layer_name in enumerate(feat_layers):
        layer_shape = feat_shapes[layer_name]
        lh, lw = layer_shape

        length = lh * lw * 8
        layer_link_gt = link_gt[idx: idx + length]
        idx = idx + length;
        layer_link_gt = np.reshape(layer_link_gt, (lh, lw, 8))
        inter_layer_link_gts[layer_name] = layer_link_gt

    for layer_idx in range(1, len(feat_layers)):
        layer_name = feat_layers[layer_idx]
        layer_shape = feat_shapes[layer_name]
        lh, lw = layer_shape
        length = lh * lw * 4
        layer_link_gt = link_gt[idx: idx + length]
        idx = idx + length
        layer_link_gt = np.reshape(layer_link_gt, (lh, lw, 4))
        cross_layer_link_gts[layer_name] = layer_link_gt


    print(idx)
    print(len(link_gt))
    assert idx == len(link_gt)


    return inter_layer_link_gts, cross_layer_link_gts

def group_segs(seg_scores, link_scores, seg_conf_threshold, link_conf_threshold,feat_layers,feat_shapes):
    """
    传入应该被resize到一维
    group segments based on their scores and links.
    Return: segment groups as a list, consisting of list of segment indexes, reprensting a group of segments belonging to a same bbox.
    """
    assert len(np.shape(seg_scores)) == 1
    assert len(np.shape(link_scores)) == 1

    valid_segs = np.where(seg_scores >= seg_conf_threshold)[0]  # `np.where` returns a tuple

    assert valid_segs.ndim == 1
    mask = {}
    for s in valid_segs:
        mask[s] = -1
    #构建了mask字典，所有的key是大于阈值的位置，value为-1
    def get_root(idx):
        parent = mask[idx]#-1代表没有root，将原idx返回
        while parent != -1:#
            idx = parent
            parent = mask[parent]
        return idx

    def union(idx1, idx2):
        root1 = get_root(idx1)
        root2 = get_root(idx2)

        if root1 != root2:
            mask[root1] = root2#如果不一致则indx1的parent是idx2

    def to_list():
        result = {}
        for idx in mask:
            root = get_root(idx)
            if root not in result:
                result[root] = []

            result[root].append(idx)

        return [result[root] for root in result]

    seg_indexes = np.arange(len(seg_scores))#[0, ..., 4437]
    layer_seg_indexes = reshape_labels_by_layer(seg_indexes,feat_layers,feat_shapes)#将上面的数组按照各层的样式reshape

    layer_inter_link_scores, layer_cross_link_scores = reshape_link_gt_by_layer(link_scores,feat_layers,feat_shapes)
    #将link_score整形到各个layer的尺寸，返回两个字典；inter_layer中每个字典元素是一个（x,y,4）大小的，表示一个xy大小的矩阵和其4领域的link值
    #各层大小依次为(16,16,4);(8,8,4)...

    for layer_index, layer_name in enumerate(feat_layers):
        layer_shape = feat_shapes[layer_name]
        lh, lw = layer_shape
        layer_seg_index = layer_seg_indexes[layer_name]#这一层对应大小的矩阵，内容为序号，conv4_3为0到64×64
        layer_inter_link_score = layer_inter_link_scores[layer_name]#取出这一层的得分
        if layer_index > 0:#如果不是第一层则应该有cross_layer score
            previous_layer_name = feat_layers[layer_index - 1]
            previous_layer_seg_index = layer_seg_indexes[previous_layer_name]
            previous_layer_shape = feat_shapes[previous_layer_name]
            plh, plw = previous_layer_shape
            layer_cross_link_score = layer_cross_link_scores[layer_name]

        for y in range(lh):
            for x in range(lw):#对于每个像素点
                seg_index = layer_seg_index[y, x]#取其标号
                _seg_score = seg_scores[seg_index]#取其得分
                if _seg_score >= seg_conf_threshold:#如果大于阈值

                    # find inter layer linked neighbours
                    inter_layer_neighbours = get_inter_layer_neighbours(x, y)#找到其本层的邻居，返回一个列表，每个元素是个元组
                    for nidx, nxy in enumerate(inter_layer_neighbours):
                        nx, ny = nxy#取其横纵坐标

                        # 如果该坐标不是空值，这个点邻居的置信度大于阈值（这个segment也在预测列表中），这两个之间的link置信度也大于阈值
                        if is_valid_cord(nx, ny, lw, lh) and \
                                seg_scores[layer_seg_index[ny, nx]] >= seg_conf_threshold and \
                                layer_inter_link_score[y, x, nidx] >= link_conf_threshold:
                            n_seg_index = layer_seg_index[ny, nx]#取出这个segment的标号
                            union(seg_index, n_seg_index)#合并其和其邻居的segment标号

                    # find cross layer linked neighbours
                    if layer_index > 0:
                        cross_layer_neighbours = get_cross_layer_neighbours(x, y)
                        for nidx, nxy in enumerate(cross_layer_neighbours):
                            nx, ny = nxy
                            if is_valid_cord(nx, ny, plw, plh) and \
                                    seg_scores[previous_layer_seg_index[ny, nx]] >= seg_conf_threshold and \
                                    layer_cross_link_score[y, x, nidx] >= link_conf_threshold:
                                n_seg_index = previous_layer_seg_index[ny, nx]
                                union(seg_index, n_seg_index)
    return to_list()

def decode_seg_offsets_pred(seg_offsets_pred,anchors):

    prior_scaling = [0.2, 0.5, 0.2, 0.5, 20.0]

    anchor_cx, anchor_cy, anchor_w, anchor_h = (anchors[:, idx] for idx in range(4))
    offset_cx = seg_offsets_pred[:, 0] * prior_scaling[0]
    offset_cy = seg_offsets_pred[:, 1] * prior_scaling[1]
    offset_w = seg_offsets_pred[:, 2] * prior_scaling[2]
    offset_h = seg_offsets_pred[:, 3] * prior_scaling[3]
    offset_theta = seg_offsets_pred[:, 4] * prior_scaling[4]

    seg_cx = anchor_cx + anchor_w * offset_cx
    seg_cy = anchor_cy + anchor_h * offset_cy  # anchor_h == anchor_w
    seg_w = anchor_w * np.exp(offset_w)
    seg_h = anchor_h * np.exp(offset_h)
    seg_theta = offset_theta

    seg_loc = np.transpose(np.vstack([seg_cx, seg_cy, seg_w, seg_h, seg_theta]))
    return seg_loc

def combine_segs(segs, return_bias=False):

    def sin(theta):
        return np.sin(theta / 180.0 * np.pi)

    def cos(theta):
        return np.cos(theta / 180.0 * np.pi)

    def tan(theta):
        return np.tan(theta / 180.0 * np.pi)

    segs = np.asarray(segs)
    assert segs.ndim == 2
    assert segs.shape[-1] == 5

    if len(segs) == 1:
        return segs[0, :]

    # find the best straight line fitting all center points: y = kx + b
    cxs = segs[:, 0]
    cys = segs[:, 1]

    ## the slope
    bar_theta = np.mean(segs[:, 4])  # average theta
    k = tan(bar_theta)

    ## the bias: minimize sum (k*x_i + b - y_i)^2
    ### let c_i = k*x_i - y_i
    ### sum (k*x_i + b - y_i)^2 = sum(c_i + b)^2
    ###                           = sum(c_i^2 + b^2 + 2 * c_i * b)
    ###                           = n * b^2 + 2* sum(c_i) * b + sum(c_i^2)
    ### the target b = - sum(c_i) / n = - mean(c_i) = mean(y_i - k * x_i)
    b = np.mean(cys - k * cxs)

    # find the projections of all centers on the straight line
    ## firstly, move both the line and centers upward by distance b, so as to make the straight line crossing the point(0, 0): y = kx
    ## reprensent the line as a vector (1, k), and the projection of vector(x, y) on (1, k) is: proj = (x + k * y)  / sqrt(1 + k^2)
    ## the projection point of (x, y) on (1, k) is (proj * cos(theta), proj * sin(theta))
    t_cys = cys - b
    projs = (cxs + k * t_cys) / np.sqrt(1 + k ** 2)
    proj_points = np.transpose([projs * cos(bar_theta), projs * sin(bar_theta)])

    # find the max distance
    max_dist = -1
    idx1 = -1
    idx2 = -1

    for i in range(len(proj_points)):
        point1 = proj_points[i, :]
        for j in range(i + 1, len(proj_points)):
            point2 = proj_points[j, :]
            dist = np.sqrt(np.sum((point1 - point2) ** 2))
            if dist > max_dist:
                idx1 = i
                idx2 = j
                max_dist = dist
    assert idx1 >= 0 and idx2 >= 0
    # the bbox: bcx, bcy, bw, bh, average_theta
    seg1 = segs[idx1, :]
    seg2 = segs[idx2, :]
    bcx, bcy = (seg1[:2] + seg2[:2]) / 2.0
    bh = np.mean(segs[:, 3])
    bw = max_dist + (seg1[2] + seg2[2]) / 2.0

    if return_bias:
        return bcx, bcy, bw, bh, bar_theta, b  # bias is useful for debugging.
    else:
        return bcx, bcy, bw, bh, bar_theta

def bboxes_to_xys(bboxes, image_shape):
    """Convert Seglink bboxes to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    if len(bboxes) == 0:
        return []

    assert np.ndim(bboxes) == 2 and np.shape(bboxes)[-1] == 5, 'invalid `bboxes` param with shape =  ' + str(
        np.shape(bboxes))

    h, w = image_shape[0:2]

    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x

    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y

    xys = np.zeros((len(bboxes), 8))
    for bbox_idx, bbox in enumerate(bboxes):
        bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox[4])
        points = cv2.boxPoints(bbox)
        points = np.int0(points)
        for i_xy, (x, y) in enumerate(points):
            x = get_valid_x(x)
            y = get_valid_y(y)
            points[i_xy, :] = [x, y]
        points = np.reshape(points, -1)
        xys[bbox_idx, :] = points
    return xys

def seglink2bbox(seg_scores, link_scores, seg_offsets_pred,
                    image_shape, seg_conf_threshold, link_conf_threshold):

    feat_layers = ["conv4_3", "fc7", "conv6_2", "conv7_2", "conv8_2", "conv9_2"]
    feat_shapes = {"conv4_3": (64, 64), "fc7": (32, 32), "conv6_2": (16, 16),
                       "conv7_2": (8, 8), "conv8_2": (4, 4), "conv9_2": (2, 2)}
    ref_h, ref_w = image_shape
    all_anchors = []

    for layer in feat_layers:
        Feat_height, Feat_width = feat_shapes[layer]
        anchors = _generate_anchors_one_layer(ref_h, ref_w, Feat_height, Feat_width)
        all_anchors.append(anchors)
    all_anchors = [np.reshape(t, (-1, t.shape[-1])) for t in all_anchors]
    all_anchors = np.vstack(all_anchors)#(4437,5)

    seg_groups = group_segs(seg_scores, link_scores, seg_conf_threshold, link_conf_threshold,feat_layers,feat_shapes)
    seg_locs = decode_seg_offsets_pred(seg_offsets_pred,all_anchors)

    bboxes = []

    for group in seg_groups:
        group = [seg_locs[idx, :] for idx in group]
        bbox = combine_segs(group)
        image_h, image_w = image_shape[0:2]
        scale = [image_w * 1.0 / ref_w, image_h * 1.0 / ref_h, image_w * 1.0 / ref_w, image_h * 1.0 / ref_h, 1]
        bbox = np.asarray(bbox) * scale
        bboxes.append(bbox)

    bboxes = bboxes_to_xys(bboxes, image_shape)
    return np.asarray(bboxes, dtype=np.float32)