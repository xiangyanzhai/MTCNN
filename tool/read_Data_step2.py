# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
from sklearn.externals import joblib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def crop_img_bboxes(img, bboxes, config):
    h, w = img.shape[:2]
    ori_img = img
    ori_bboxes = bboxes
    bboxes = bboxes[:, :4]
    jitter = np.random.choice(config.jitter_ratio)
    a = int(h * jitter)
    b = int(w * jitter)
    h1 = np.random.randint(a)
    h2 = np.random.randint(a - h1)
    w1 = np.random.randint(b)
    w2 = np.random.randint(b - w1)

    h2 = h - h2
    w2 = w - w2
    img = img[h1:h2, w1:w2]

    x1 = np.maximum(w1, bboxes[:, 0:1])
    y1 = np.maximum(h1, bboxes[:, 1:2])
    x2 = np.minimum(w2 - 1, bboxes[:, 2:3])
    y2 = np.minimum(h2 - 1, bboxes[:, 3:4])

    x1 = x1 - w1
    y1 = y1 - h1
    x2 = x2 - w1
    y2 = y2 - h1

    areas1 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    w = np.maximum(0., x2 - x1)
    h = np.maximum(0., y2 - y1)
    areas2 = w * h
    areas2 = areas2.ravel()

    inds = (areas2 / areas1) >= 1.

    bboxes = np.concatenate([x1, y1, x2, y2], axis=1)
    bboxes = bboxes[inds]

    if bboxes.shape[0] == 0 or np.random.random() < config.keep_ratio:
        return ori_img, ori_bboxes
    t = ori_bboxes[:, 4:]
    t = t[inds]
    if t.sum() > 0:
        t = t.reshape(-1, 5, 2)
        t = t - np.array([w1, h1])
        t = t.reshape(-1, 10)
    bboxes = np.concatenate((bboxes, t), axis=-1)
    return img, bboxes


class Read_Data(Dataset):
    def __init__(self, config):
        self.files = []
        for i in config.files:
            self.files += joblib.load(i)
        self.config = config

        self.transform = transforms.Compose([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_file, bboxes, anchors = self.files[index]
        bboxes = bboxes.copy()
        img = cv2.imread(img_file)

        H, W = img.shape[:2]

        if np.random.random() > 0.5:
            img = img[:, ::-1].copy()
            bboxes = self.bboxes_left_right(bboxes, img.shape[1])
            anchors = self.bboxes_left_right(anchors, img.shape[1])

        if bboxes.shape[0] == 0:  # 当图片中没有目标时，填充一个bboxes，训练时在get_loss中再过滤掉label<0的，只保留label>=0
            bboxes = np.array([[W / 4., H / 4., W * 3 / 4., H * 3 / 4., -100, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              dtype=np.float32)
        if np.random.random() > 0.2:
            img = Image.fromarray(img[..., ::-1])
            img = self.transform(img)
            img = np.array(img)
            img = np.ascontiguousarray(img[..., ::-1])
        # if np.random.random()>0.5:
        #     img, bboxes= crop_img_bboxes(img, bboxes,self.config)

        img = torch.tensor(img)
        bboxes = torch.tensor(bboxes)
        anchors = torch.tensor(anchors)
        return img, bboxes, bboxes.shape[0], img.shape[0], img.shape[1], anchors, anchors.shape[0]

    def bboxes_left_right(self, bboxes, w):
        bboxes[:, 0], bboxes[:, 2] = w - 1. - bboxes[:, 2], w - 1. - bboxes[:, 0]

        if bboxes[:, 4:].sum() > 0:
            bboxes[:, 4], bboxes[:, 6] = w - 1. - bboxes[:, 6], w - 1. - bboxes[:, 4]
            bboxes[:, 8] = w - 1. - bboxes[:, 8]
            bboxes[:, 10], bboxes[:, 12] = w - 1. - bboxes[:, 12], w - 1. - bboxes[:, 10]
        return bboxes


def draw_gt(im, gt, color=(0, 255, 255)):
    im = im.astype(np.uint8).copy()
    boxes = gt.astype(np.int32)
    print(im.max(), im.min(), im.shape)
    for box in boxes:
        # print(box)
        x1, y1, x2, y2 = box[:4]

        im = cv2.rectangle(im, (x1, y1), (x2, y2), color)

    im = im.astype(np.uint8)
    cv2.imshow('a', im)
    cv2.waitKey(2000)
    return im


def func(batch):
    return batch
    # # imgs, bboxes, num_bbox, H, W = zip(*batch)
    # m = len(batch)
    # num_bbox = []
    # H = []
    # W = []
    # for i in range(m):
    #     num_bbox.append(batch[i][-3])
    #     H.append(batch[i][-2])
    #     W.append(batch[i][-1])
    # max_num_b = max(num_bbox)
    # max_H = max(H)
    # max_W = max(W)
    #
    # new_img = np.zeros((m, max_H, max_W, 3), dtype=np.uint8)
    # new_bboxes = np.zeros((m, max_num_b, 5), dtype=np.float32)
    # for i in range(m):
    #     new_img[i][:H[i], :W[i]] = batch[i][0]
    #     new_bboxes[i][:num_bbox[i]] = batch[i][1]
    #
    # new_img = torch.from_numpy(new_img)
    # new_bboxes = torch.from_numpy(new_bboxes)
    #
    # num_bbox = torch.tensor(num_bbox)
    # H = torch.tensor(H)
    # W = torch.tensor(W)
    #
    # return new_img, new_bboxes, num_bbox, H, W
    #


def func(batch):
    m = len(batch)
    num_b = []
    num_H = []
    num_W = []
    num_A = []
    for i in range(m):
        num_b.append(batch[i][2])
        num_H.append(batch[i][3])
        num_W.append(batch[i][4])
        num_A.append(batch[i][-1])

    max_b = max(num_b)
    max_H = max(num_H)
    max_W = max(num_W)
    max_A = max(num_A)
    imgs = []
    bboxes = []
    anchors = []
    for i in range(m):
        imgs.append(batch[i][0].resize_(max_H, max_W, 3)[None])
        bboxes.append(batch[i][1].resize_(max_b, 14)[None])
        anchors.append(batch[i][-2].resize_(max_A, 4)[None])

    imgs = torch.cat(imgs, dim=0)
    bboxes = torch.cat(bboxes, dim=0)
    anchors = torch.cat(anchors, dim=0)
    return imgs, bboxes, torch.tensor(num_b, dtype=torch.int64), torch.tensor(num_H, dtype=torch.int64), torch.tensor(
        num_W, dtype=torch.int64), anchors, torch.tensor(num_A, dtype=torch.int64)


def handel(x, bboxes, num_b, H, W, tanchors, num_A):
    x = x.view(-1)[:H * W * 3].view(H, W, 3)
    bboxes = bboxes[:num_b]
    inds = bboxes[:, 5] >= 0
    bboxes = bboxes[inds]
    tanchors = tanchors[:num_A]
    return x, bboxes, tanchors


if __name__ == "__main__":
    from datetime import datetime
    from MTCNN.tool.config import Config

    path = '/home/zhai/PycharmProjects/Demo35/MTCNN/data_process/'

    files = [path + 'O_net_train_Celeba_align.pkl']
    config = Config(files, lr=0.001, weight_decay=0.0000, scale=0.5 ** 0.5,

                    batch_size_per_GPU=8, gpus=1,
                    P_net_conf_thresh=0.5,
                    P_net_iou_thresh=0.7,
                    R_net_conf_thresh=0.5,
                    R_net_iou_thresh=0.7,
                    O_net_conf_thresh=0.5,
                    O_net_iou_thresh=0.7,

                   )
    dataset = Read_Data(config)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=16, collate_fn=func)
    c = 0
    z = 0
    for i in range(2):
        for imgs, bboxes, num_b, num_H, num_W, anchors, num_A in dataloader:
            for j in range(imgs.shape[0]):
                x, bbox, tanchors = handel(imgs[j], bboxes[j], num_b[j], num_H[j], num_W[j], anchors[j], num_A[j])
                print(x.shape, bbox.shape, tanchors.shape)
                x = x.cpu().numpy()
                bbox = bbox.cpu().numpy()
                tanchors = tanchors.cpu().numpy()
                im = draw_gt(x, bbox, (0,255, 255))
                im = draw_gt(im, tanchors)

        print(i)

        # print(len(x))
        # img, bboxes = x[0][:2]
        # c += 1
        # print(datetime.now(), c)
        # draw_gt(img.numpy(), bboxes.numpy())
