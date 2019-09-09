# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import torch

is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


Zero = cuda(torch.Tensor([0]))


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
    areas1 = hw.prod(dim=-1)
    hw = bboxes[:, 2:4] - bboxes[:, :2] + 1
    areas2 = hw.prod(dim=-1)

    yx1 = torch.max(pre_bboxes[:, None, :2], bboxes[:, :2])
    yx2 = torch.min(pre_bboxes[:, None, 2:4], bboxes[:, 2:4])

    hw = yx2 - yx1 + 1

    hw = torch.max(hw, Zero)
    areas_i = hw.prod(dim=-1)

    if mode == "Union":
        iou = areas_i / (areas1[:, None] + areas2 - areas_i)
    elif mode == "Minimum":
        iou = areas_i.view(-1)/ (torch.min(areas1, areas2))

    return iou


def nms(bboxes, thresh, mode='Union'):
    index = torch.arange(bboxes.shape[0])
    keep = []

    while index.shape[0] > 0:
        keep.append(index[0])

        iou = cal_IOU(bboxes[index], bboxes[index[:1]],mode=mode)

        iou = iou.view(-1, )
        inds = iou <= thresh

        index = index[inds]

    return torch.tensor(keep)


if __name__ == "__main__":
    pass
