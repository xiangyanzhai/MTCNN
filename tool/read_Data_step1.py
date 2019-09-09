# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
from sklearn.externals import joblib
import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image


def crop_img(img, bboxes):
    img_h, img_w = img.shape[:2]
    if bboxes.shape[0] == 0:
        n_w, n_h = min(516, img_w), min(516, img_h)
        t_x, t_y = np.random.choice(int(img_w - n_w)), np.random.choice(int(img_h - n_h))
        t_x, t_y = max(t_x, 0), max(t_y, 0)
        t_x2, t_y2 = t_x + 516, t_y + 516
        return img[t_x:t_x2, t_y:t_y2], bboxes
    hw = bboxes[:, 2:4] - bboxes[:, :2] + 1
    areas = hw.prod(axis=-1).ravel()
    index = np.random.choice(bboxes.shape[0])
    temp = bboxes[index]
    x1, y1, x2, y2 = temp[:4]
    w, h = x2 - x1 + 1, y2 - y1 + 1

    n_w, n_h = max(516, w + 128), max(516, h + 128)
    t_x1, t_y1 = x1 - np.random.choice(max(int(n_w - w - 64), 1)) - 32, y1 - np.random.choice(
        max(int(n_h - h - 64), 1)) - 32
    t_x1, t_y1 = max(t_x1, 0), max(t_y1, 0)
    t_x2, t_y2 = t_x1 + n_w, t_y1 + n_w
    t_x2, t_y2 = min(t_x2, img_w), min(t_y2, img_h)
    t_x1, t_x2, t_y1, t_y2 = int(t_x1), int(t_x2), int(t_y1), int(t_y2)
    n_img = img[t_y1:t_y2, t_x1:t_x2]
    img_h, img_w = n_img.shape[:2]

    bboxes[:, :4] = bboxes[:, :4] - np.array([t_x1, t_y1, t_x1, t_y1])
    bboxes[..., slice(0, 4, 2)] = np.clip(bboxes[..., slice(0, 4, 2)], 0, img_w - 1)
    bboxes[..., slice(1, 4, 2)] = np.clip(bboxes[..., slice(1, 4, 2)], 0, img_h - 1)
    hw = bboxes[:, 2:4] - bboxes[:, :2] + 1
    areas2 = hw.prod(axis=-1).ravel()
    t = areas2 / areas
    t = t > 0.7
    bboxes = bboxes[t]
    return n_img, bboxes


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
        img_file, bboxes = self.files[index]
        bboxes = bboxes.copy()
        img = cv2.imread(img_file)

        img, bboxes = crop_img(img, bboxes)

        H, W = img.shape[:2]

        if np.random.random() > 0.5:
            img = img[:, ::-1].copy()
            bboxes = self.bboxes_left_right(bboxes, img.shape[1])

        if bboxes.shape[0] == 0:  # 当图片中没有目标时，填充一个bboxes，训练时在get_loss中再过滤掉label<0的，只保留label>=0
            bboxes = np.array([[W / 4., H / 4., W * 3 / 4., H * 3 / 4., -100, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              dtype=np.float32)

        if np.random.random() > 0.2:
            img = Image.fromarray(img[..., ::-1])
            img = self.transform(img)
            img = np.array(img)
            img = np.ascontiguousarray(img[..., ::-1])

        # if np.random.random()>0.2:
        #     img, bboxes= crop_img_bboxes(img, bboxes,self.config)
        # if np.random.random()>0.8:
        #     img,bboxes=rotate(img,bboxes)
        img = torch.tensor(img)
        bboxes = torch.tensor(bboxes)
        return img, bboxes, bboxes.shape[0], img.shape[0], img.shape[1]

    def bboxes_left_right(self, bboxes, w):
        bboxes[:, 0], bboxes[:, 2] = w - 1. - bboxes[:, 2], w - 1. - bboxes[:, 0]

        if bboxes[:, 4:].sum() > 0:
            bboxes[:, 4], bboxes[:, 6] = w - 1. - bboxes[:, 6], w - 1. - bboxes[:, 4]
            bboxes[:, 8] = w - 1. - bboxes[:, 8]
            bboxes[:, 10], bboxes[:, 12] = w - 1. - bboxes[:, 12], w - 1. - bboxes[:, 10]
        return bboxes


def draw_gt(im, gt):
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
    # new_img = np.zeros((m, max_H, max_W, 3)1, dtype=np.uint8)
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


if __name__ == "__main__":
    from datetime import datetime
    from datetime import datetime
    from MTCNN.tool.config import Config

    path = '/home/zhai/PycharmProjects/Demo35/MTCNN/data_process/'

    files = [path + 'wider.pkl']

    config = Config(files, lr=0.001, weight_decay=0.0000, scale=0.5 ** 0.5,
                    batch_size_per_GPU=8, gpus=1,
                    P_net_conf_thresh=0.5,
                    P_net_iou_thresh=0.7,
                    R_net_conf_thresh=0.5,
                    R_net_iou_thresh=0.7,
                    O_net_conf_thresh=0.5,
                    O_net_iou_thresh=0.7,
                    roi_min_size=12,
                    )
    dataset = Read_Data(config)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, collate_fn=lambda x: x)
    c = 0
    for i in range(2):

        for x in dataloader:
            img, bboxes = x[0][:2]
            c += 1
            # h,w=img.shape[:2]
            # if h<12 or w<12:

            print(datetime.now(), c)
            if bboxes.shape[0] == 0:
            #     print('aaaaaaaaaaaaaaa')
                draw_gt(img.numpy(), bboxes.numpy())

            draw_gt(img.numpy(), bboxes.numpy())
