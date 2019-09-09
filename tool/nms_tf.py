# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np

import tensorflow as tf

thresh = None


def py_nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        # keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def cal_IOU(pre_bboxes, bboxes, mode="Union"):
    hw = pre_bboxes[:, 2:4] - pre_bboxes[:, :2]
    areas1 = tf.reduce_prod(hw, axis=-1)

    hw = bboxes[:, 2:4] - bboxes[:, :2] + 1
    areas2 = tf.reduce_prod(hw, axis=-1)

    yx1 = tf.maximum(pre_bboxes[:, None, :2], bboxes[:, :2])
    yx2 = tf.minimum(pre_bboxes[:, None, 2:4], bboxes[:, 2:4])

    hw = yx2 - yx1 + 1
    hw = tf.maximum(hw, 0)
    areas_i = tf.reduce_prod(hw, axis=-1)
    if mode == "Union":
        iou = areas_i / (areas1[:, None] + areas2 - areas_i)
    else:
        iou = tf.reshape(areas_i, (-1,)) / tf.minimum(areas1, areas2)
    return iou


def cond(m, index, keep, bboxes):
    return m > 0


def body(m, index, keep, bboxes):
    keep = tf.concat([keep, index[:1]], axis=0)
    iou = cal_IOU(bboxes, bboxes[:1], mode='Minimum')
    inds = iou <= thresh
    bboxes = tf.boolean_mask(bboxes, inds)
    index = tf.boolean_mask(index, inds)
    return tf.shape(bboxes)[0], index, keep, bboxes

    pass


def nms(bboxes, thresh_):
    global thresh
    thresh = thresh_
    index = tf.range(tf.shape(bboxes)[0], dtype=tf.int32)
    keep = tf.zeros([0], dtype=tf.int32)
    m = tf.shape(bboxes)[0]
    _, _, keep, _ = tf.while_loop(cond, body, [m, index, keep, bboxes],
                                  shape_invariants=[tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None]),
                                                    tf.TensorShape([None, 4])], back_prop=False)

    return keep


if __name__ == "__main__":
    pass
