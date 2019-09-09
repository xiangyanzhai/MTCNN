# !/usr/bin/python
# -*- coding:utf-8 -*-
import torch

is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


def cal_IOU(pre_bboxes, bboxes):
    hw = pre_bboxes[:, 2:4] - pre_bboxes[:, :2]
    areas1 = hw.prod(dim=-1)
    hw = bboxes[:, 2:4] - bboxes[:, :2]
    areas2 = hw.prod(dim=-1)

    yx1 = torch.max(pre_bboxes[:, None, :2], bboxes[:, :2])
    yx2 = torch.min(pre_bboxes[:, None, 2:4], bboxes[:, 2:4])

    hw = yx2 - yx1

    hw = torch.max(hw, cuda(torch.Tensor([0])))
    areas_i = hw.prod(dim=-1)
    iou = areas_i / (areas1[:, None] + areas2 - areas_i)
    return iou


def bbox2loc(anchor, bbox):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]

    hw = bbox[..., 2:4] - bbox[..., 0:2]

    t_yx = (bbox[..., :2] - anchor[..., :2]) / c_hw
    t_hw = hw / c_hw
    return torch.cat([t_yx, t_hw], dim=1)


class AnchorTargetCreator(object):
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        if bbox.shape[0] == 0 or anchor.shape[0] == 0:
            inds = torch.randperm(anchor.shape[0])[:3]
            label = cuda(torch.zeros(inds.shape[0], dtype=torch.int64))
            indsP = cuda(torch.tensor([], dtype=torch.int64))
            loc = cuda(torch.zeros((0, 4), dtype=torch.float32))
            return inds, label, indsP, loc
        index_inside = torch.arange(anchor.shape[0])
        # index_inside = index_inside[
        #     (anchor[:, 0] >= 0) & (anchor[:, 2] <= img_size[1]) & (anchor[:, 1] >= 0) & (anchor[:, 3] <= img_size[0])]

        # index_inside = index_inside[(anchor[:, 0] <= img_size[1] - 6) & (anchor[:, 1] <= img_size[0] - 6)]

        anchor = anchor[index_inside]
        IOU = cal_IOU(anchor, bbox)
        iou, inds_box = IOU.max(dim=1)

        indsP = iou >= 0.65
        indsN = iou < 0.3
        indsM = (iou >= 0.4) & (iou < 0.65)

        t = torch.arange(indsP.shape[0])
        indsP = t[indsP]
        indsN = t[indsN]
        indsM = t[indsM]
        p_num = indsP.shape[0]
        n_num = indsN.shape[0]
        m_num = indsM.shape[0]
        # print('a', p_num, m_num, n_num)
        n_pos = min(min(p_num, m_num), int(n_num / 3))
        n_pos = max(n_pos, 1)

        n_mid = n_pos
        n_neg = n_pos * 3
        n_neg = max(n_neg, 32)
        # print('b', n_pos, n_mid, n_neg)
        indsP = indsP[torch.randperm(p_num)[:n_pos]]
        indsN = indsN[torch.randperm(n_num)[:n_neg]]
        indsM = indsM[torch.randperm(m_num)[:n_mid]]
        n_pos = indsP.shape[0]
        n_mid = indsM.shape[0]
        n_neg = indsN.shape[0]
        # print('c', n_pos, n_mid, n_neg)
        inds_reg = torch.cat([indsP, indsM], dim=0)
        anchor = anchor[inds_reg]

        bbox = bbox[inds_box[inds_reg]]

        loc = bbox2loc(anchor, bbox)

        label = cuda(torch.zeros(n_pos + n_mid + n_neg, dtype=torch.int64))
        label[:n_pos] = 1
        label[n_pos:n_pos + n_mid] = 2
        inds = torch.cat([indsP, indsM, indsN], dim=0)
        inds = index_inside[inds]
        inds_reg = index_inside[inds_reg]
        return inds, label, inds_reg, loc
