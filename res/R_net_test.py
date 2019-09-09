# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch

is_gpu = torch.cuda.is_available()


def cuda(x):
    if is_gpu:
        x = x.cuda()
    return x


import numpy as np
import cv2
import codecs
from sklearn.externals import joblib
import torch.nn.functional as F
from maskrcnn_benchmark.layers import nms as _box_nms
from maskrcnn_benchmark.layers import ROIAlign
from MTCNN.tool.P_net import P_net
from MTCNN.tool.R_net import R_net
from MTCNN.tool.config import Config
from datetime import datetime

roialign_24 = ROIAlign((24, 24), 1 / 1., 2)


def loc2bbox(pre_loc, anchor):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    hw = pre_loc[..., 2:4] * c_hw
    yx1 = anchor[..., :2] + pre_loc[..., :2] * c_hw
    yx2 = yx1 + hw
    bboxes = torch.cat((yx1, yx2), dim=-1)
    return bboxes


def bbox2square(bboxes):
    if bboxes.shape[0] > 0:
        wh = bboxes[:, 2:4] - bboxes[:, :2]
        wh, _ = wh.max(dim=-1)
        wh = wh[:, None]
        xy = (bboxes[:, 2:4] + bboxes[:, :2]) / 2
        xy1 = xy - wh / 2
        xy1 = torch.max(xy1, cuda(torch.Tensor([0])))
        xy2 = xy1 + wh
        bboxes = torch.cat([xy1, xy2], dim=-1)
    return bboxes


def get_xy(N):
    t = np.arange(N)
    x, y = np.meshgrid(t, t)
    x = x[..., None]
    y = y[..., None]
    xy = np.concatenate((x, y), axis=-1)

    return xy


def get_anchors(N):
    xy = get_xy(N) * 2
    xy2 = xy + 12
    anchors = np.concatenate((xy, xy2), axis=-1)
    anchors = anchors.astype(np.float32)

    return torch.tensor(anchors)


class P_net_test():
    def __init__(self, config):
        self.config = config
        self.scale = config.scale
        self.anchors = cuda(get_anchors(5000))
        self.P_net = P_net()
        self.R_net = R_net()

    def get_res(self, x):
        x = cuda(x.float())
        x = x[None]
        x = x.permute(0, 3, 1, 2)

        tanchors = self.get_P_net_res(x)
        if tanchors.shape[0] == 0:
            return tanchors
        res = self.get_R_net_res(x, tanchors[..., :4])

        return res

    def get_R_net_res(self, x, tanchors):
        h, w = x.shape[2:]
        tanchors = bbox2square(tanchors[..., :4])
        if tanchors.shape[0] > 0:
            t, _ = tanchors.max(dim=0)
            t = t + 1
            t = t.long()
            n_w, n_h = t[2], t[3]
            x = F.pad(x, (0, max(n_w - w, 0), 0, max(n_h - h, 0)), mode='constant', value=127.5)

        roi = tanchors
        roi_inds = cuda(torch.zeros((roi.size()[0], 1)))
        roi = torch.cat([roi_inds, roi], dim=1)
        xx = roialign_24(x, roi)
        xx = (xx - 127.5) / 128.0
        R_net_logits, R_net_loc, R_net_landmarks = self.R_net(xx)
        # R_net_logits = R_net_logits.permute(0, 2, 3, 1).view(-1, 2)
        # R_net_loc = R_net_loc.permute(0, 2, 3, 1).view(-1, 4)
        # R_net_landmarks = R_net_landmarks.permute(0, 2, 3, 1).view(-1, 10)

        score = F.softmax(R_net_logits, dim=-1)[..., 1]

        inds = score >= self.config.R_net_conf_thresh
        if inds.sum() == 0:
            return cuda(torch.zeros((0, 5)))
        score = score[inds]
        net_loc = R_net_loc[inds]
        net_landmarks = R_net_landmarks[inds]
        tanchors = tanchors[inds]

        bboxes = loc2bbox(net_loc, tanchors)

        bboxes[..., slice(0, 4, 2)] = torch.clamp(bboxes[..., slice(0, 4, 2)], 0, w - 1)
        bboxes[..., slice(1, 4, 2)] = torch.clamp(bboxes[..., slice(1, 4, 2)], 0, h - 1)
        hw = bboxes[:, 2:4] - bboxes[:, :2]
        inds = hw >= self.config.roi_min_size[1]
        inds = inds.all(dim=1)
        bboxes = bboxes[inds]
        score = score[inds]
        if bboxes.shape[0] == 0:
            return cuda(torch.zeros((0, 5)))

        score, inds = score.sort(descending=True)
        bboxes = bboxes[inds]
        keep = _box_nms(bboxes, score, self.config.R_net_iou_thresh)

        bboxes = bboxes[keep]
        score = score[keep]
        score = score.view(-1, 1)

        return torch.cat([bboxes, score], dim=-1)

    def get_P_net_res(self, x):

        h, w = x.shape[2:]

        roi = cuda(torch.tensor([[0, 0, 0, w, h]]).float())
        i = 0
        all_bboxes = []
        all_score = []
        while True:
            n_h, n_w = int(h * self.scale ** i), int(w * self.scale ** i)
            if n_h < 12 or n_w < 12:
                break

            roialign = ROIAlign((n_h, n_w), 1 / 1., 2)
            xx = roialign(x, roi)

            # xx = F.interpolate(x, size=(n_h, n_w))

            a = np.ceil(n_h / 12.) * 12
            b = np.ceil(n_w / 12.) * 12
            a = int(a)
            b = int(b)
            xx = F.pad(xx, (0, b - n_w, 0, a - n_h), mode='constant', value=127.5)
            xx = (xx - 127.5) / 128.0

            P_net_logits, P_net_loc, P_net_landmarks = self.P_net(xx)

            map_H, map_W = P_net_logits.shape[2:]
            P_net_logits = P_net_logits.permute(0, 2, 3, 1).contiguous().view(-1, 2)
            P_net_loc = P_net_loc.permute(0, 2, 3, 1).contiguous().view(-1, 4)
            P_net_landmarks = P_net_landmarks.permute(0, 2, 3, 1).contiguous().view(-1, 10)
            anchors = self.anchors[:map_H, :map_W].contiguous().view(-1, 4) / self.scale ** i
            i += 1

            score = F.softmax(P_net_logits, dim=-1)[..., 1]
            inds = score >= self.config.P_net_conf_thresh
            if inds.sum() == 0:
                continue

            score = score[inds]
            P_net_loc = P_net_loc[inds]

            anchors = anchors[inds]
            bboxes = loc2bbox(P_net_loc, anchors)
            bboxes[..., slice(0, 4, 2)] = torch.clamp(bboxes[..., slice(0, 4, 2)], 0, w - 1)
            bboxes[..., slice(1, 4, 2)] = torch.clamp(bboxes[..., slice(1, 4, 2)], 0, h - 1)

            hw = bboxes[..., 2:4] - bboxes[..., :2]
            inds = hw >= self.config.roi_min_size[0]
            inds = inds.all(dim=-1)
            if inds.sum() == 0:
                continue

            bboxes = bboxes[inds]
            score = score[inds]

            score, inds = score.sort(descending=True)
            bboxes = bboxes[inds]
            keep = _box_nms(bboxes, score, 0.5)
            score = score[keep]
            bboxes = bboxes[keep]

            all_bboxes.append(bboxes)
            all_score.append(score)
        if len(all_bboxes) == 0:
            return cuda(torch.zeros((0, 5)))

        bboxes = torch.cat(all_bboxes, dim=0)
        score = torch.cat(all_score, dim=0)

        score, inds = score.sort(descending=True)
        bboxes = bboxes[inds]
        keep = _box_nms(bboxes, score, self.config.P_net_iou_thresh)
        bboxes = bboxes[keep]
        score = score[keep]

        return torch.cat([bboxes, score.view(-1, 1)], dim=1)

    def test(self, model_file):
        self.P_net.load_state_dict(torch.load(model_file[0], map_location='cpu'))
        self.P_net.eval()
        cuda(self.P_net)
        self.R_net.load_state_dict(
            torch.load(model_file[1], map_location='cpu'))
        self.R_net.eval()
        cuda(self.R_net)
        ellips = joblib.load('/home/zhai/PycharmProjects/Demo35/MTCNN/data_process/ellips.pkl')
        test_dir = joblib.load('/home/zhai/PycharmProjects/Demo35/MTCNN/data_process/FDDB_test.pkl')
        path = '/home/zhai/PycharmProjects/Demo35/dataset/FDDB/'
        z = 0
        for i in range(len(test_dir)):
            w_file = '/home/zhai/PycharmProjects/Demo35/dataset/FDDB/res/fold-0%d-out.txt' % (i + 1)
            if i == 9:
                w_file = '/home/zhai/PycharmProjects/Demo35/dataset/FDDB/res/fold-%d-out.txt' % (i + 1)
            with codecs.open(w_file, 'w') as f:
                for file in test_dir[i]:
                    z += 1

                    im_file = path + file + '.jpg'
                    img = cv2.imread(im_file)

                    t_img = img
                    img = torch.tensor(img)
                    with torch.no_grad():
                        res = self.get_res(img)

                    res = res.cpu()
                    res = res.detach().numpy()

                    # draw_gt(t_img, res[..., :4], ellips[file])

                    m = res.shape[0]
                    print(datetime.now(), z, m)

                    f.write(file + '\n')
                    f.write(str(m) + '\n')
                    res[..., 2:4] = res[..., 2:4] - res[..., :2]
                    for bbox in res:
                        f.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ' + str(
                            bbox[4]) + '\n')


def draw_gt(im, gt, ellips):
    im = im.astype(np.uint8).copy()
    boxes = gt.astype(np.int32)
    print(im.max(), im.min(), im.shape)
    for box in boxes:
        # print(box)
        x1, y1, x2, y2 = box[:4]
        landmarks = box[4:]
        landmarks = landmarks.reshape(-1, 2).astype(np.int32)

        for z in landmarks:
            cv2.circle(im, (z[0], z[1]), 2, (0, 0, 255), 2)

        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))

    bboxes = ellips
    for bbox in bboxes:
        ra, rb, theta, cx, cy = bbox[:5]
        cv2.ellipse(im, (cx, cy), (rb, ra), theta, 0, 360, (0, 0, 255))
    im = im.astype(np.uint8)
    cv2.imshow('a', im)
    cv2.waitKey(2000)
    return im


from sklearn.externals import joblib

if __name__ == "__main__":
    config = Config(None, lr=0.005, weight_decay=0.0001, scale=0.79,

                    batch_size_per_GPU=32, gpus=1,
                    P_net_conf_thresh=0.5,
                    P_net_iou_thresh=0.7,
                    R_net_conf_thresh=0.5,
                    R_net_iou_thresh=0.7,
                    O_net_conf_thresh=0.5,
                    O_net_iou_thresh=0.7,
                    )
    P_test = P_net_test(config)

    P_net_model_file = '/home/zhai/PycharmProjects/Demo35/MTCNN/train/models/P_net_45000_1.pth'
    R_net_model_file = '/home/zhai/PycharmProjects/Demo35/MTCNN/train/models/R_net_45000_1.pth'
    model_file = [P_net_model_file, R_net_model_file]
    P_test.test(model_file)
