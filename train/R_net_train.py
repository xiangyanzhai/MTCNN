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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from maskrcnn_benchmark.layers import ROIAlign
from MTCNN.tool.R_net import R_net
from MTCNN.tool.read_Data_step2 import Read_Data
from MTCNN.tool.torch_ATC import AnchorTargetCreator
from MTCNN.tool.config import Config
from datetime import datetime

roialign = ROIAlign((24, 24), 1 / 1., 2)
ce_loss = nn.CrossEntropyLoss(reduction='none')
mse_loss = nn.MSELoss()


def bbox2square(bboxes):
    if bboxes.shape[0] > 0:
        wh = bboxes[..., 2:4] - bboxes[..., :2]
        wh, _ = wh.max(dim=-1)
        wh = wh[:, None]
        xy = (bboxes[..., 2:4] + bboxes[..., :2]) / 2
        xy1 = xy - wh / 2
        xy1 = torch.max(xy1, cuda(torch.Tensor([0])))
        xy2 = xy1 + wh
        bboxes = torch.cat([xy1, xy2], dim=-1)
    return bboxes


class R_net_train():
    def __init__(self):
        self.ATC = AnchorTargetCreator()
        self.model = R_net()

    def bboxes2anchors(self, bboxes):
        wh = bboxes[..., 2:4] - bboxes[..., :2]
        xy = (bboxes[..., 2:4] + bboxes[..., :2]) / 2

        # area = wh.prod(dim=-1) ** 0.5
        # area = area[:, None]
        # xy1 = xy - area / 2
        # xy1 = torch.max(xy1, cuda(torch.Tensor([0])))
        # xy2 = xy + area
        # anchors1 = torch.cat([xy1, xy2], dim=-1)

        wh, _ = wh.max(dim=-1)
        wh = wh[:, None]
        xy1 = xy - wh / 2
        xy1 = torch.max(xy1, cuda(torch.Tensor([0])))
        xy2 = xy1 + wh
        anchors2 = torch.cat([xy1, xy2], dim=-1)

        # tt = (torch.rand(area.shape) - 0.5) / 2
        # tt = cuda(tt)
        # tt = tt * area
        # xy1 = anchors1[:, :2] + tt
        # xy1 = torch.max(xy1, cuda(torch.Tensor([0])))
        # xy2 = xy1 + area
        # anchors3 = torch.cat([xy1, xy2], dim=-1)
        #
        # tt = (torch.rand(area.shape) - 0.5) / 2
        # tt = cuda(tt)
        # tt = tt * wh
        # xy1 = anchors2[:, :2] + tt
        # xy1 = torch.max(xy1, cuda(torch.Tensor([0])))
        # xy2 = xy1 + wh
        # anchors4 = torch.cat([xy1, xy2], dim=-1)
        #
        # return torch.cat([anchors1, anchors2, anchors3, anchors4], dim=0)
        anchors2 = anchors2[torch.randperm(anchors2.shape[0])[:100]]
        return anchors2

    def get_train_data(self, x, bboxes, num_b, H, W, tanchors, num_A):
        x = x.view(-1)[:H * W * 3].view(H, W, 3)
        bboxes = bboxes[:num_b]
        inds = bboxes[:, 5] >= 0
        bboxes = bboxes[inds]
        tanchors = tanchors[:num_A]
        tanchors = cuda(tanchors)
        x = cuda(x.float())
        bboxes = cuda(bboxes)
        x = x[None]
        x = x.permute(0, 3, 1, 2)
        h, w = x.shape[2:]

        tanchors = bbox2square(tanchors[..., :4])
        tanchors = torch.cat([tanchors, self.bboxes2anchors(bboxes)], dim=0)

        t, _ = tanchors.max(dim=0)
        t = t + 1
        t = t.long()
        n_w, n_h = t[2], t[3]
        x = F.pad(x, (0, max(n_w - w, 0), 0, max(n_h - h, 0)), mode='constant', value=127.5)

        inds, label, inds_reg, loc = self.ATC(bboxes, tanchors, (h, w))

        roi = tanchors[inds]
        roi_inds = cuda(torch.zeros((roi.size()[0], 1)))
        roi = torch.cat([roi_inds, roi], dim=1)
        xx = roialign(x, roi)

        xx = (xx - 127.5) / 128.0
        R_net_logits, R_net_loc, R_net_landmarks = self.model(xx)
        # R_net_logits = R_net_logits.permute(0, 2, 3, 1).view(-1, 2)
        # R_net_loc = R_net_loc.permute(0, 2, 3, 1).view(-1, 4)
        # R_net_landmarks = R_net_landmarks.permute(0, 2, 3, 1).view(-1, 10)

        inds = label < 2
        a = R_net_logits[inds]
        b = label[inds]
        inds = label > 0
        c = R_net_loc[inds]
        d = loc
        return a, b, c, d


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


def get_cls_train_data(logits, label):
    indsP = label == 1
    logits_P = logits[indsP]
    label_P = label[indsP]

    indsN = label == 0
    logits_N = logits[indsN]
    label_N = label[indsN]

    inds = torch.randperm(logits_N.shape[0])[:logits_P.shape[0] * 3]

    logits_N = logits_N[inds]
    label_N = label_N[inds]

    logits = torch.cat([logits_P, logits_N], dim=0)
    label = torch.cat([label_P, label_N], dim=0)
    return logits, label


def get_cls_loss(logits, label, batch):
    cls_loss = 0
    if label.shape[0] > 0:
        cls_loss = ce_loss(logits, label)

        cls_loss, _ = cls_loss.sort(descending=True)

        cls_loss = cls_loss[:int(cls_loss.shape[0] * 0.7) + 1]
        cls_loss = cls_loss[:batch]
        cls_loss = cls_loss.mean()
    return cls_loss


def get_bbox_loss(loc_train, loc, batch):
    bbox_loss = 0

    if loc.shape[0] > 0:
        bbox_loss = (loc_train - loc) ** 2
        bbox_loss = bbox_loss.sum(dim=-1)

        # bbox_loss = bbox_loss[torch.randperm(bbox_loss.shape[0])]
        # bbox_loss = bbox_loss[:128]
        # bbox_loss = bbox_loss.mean()

        bbox_loss, _ = bbox_loss.sort(descending=True)
        bbox_loss = bbox_loss[:int(bbox_loss.shape[0] * 0.7) + 1]
        bbox_loss = bbox_loss[:batch]
        bbox_loss = bbox_loss.mean()
    return bbox_loss


def train(P_train, config, step, x, model_file=None):
    model = P_train.model

    cuda(model)

    train_params = list(model.parameters())
    if step > 0:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))

    bias_p = []
    weight_p = []
    print(len(train_params))
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p.append(p)
        else:
            weight_p.append(p)
    print(len(weight_p), len(bias_p))
    lr = config.lr
    # if step >= int(60000 * x):
    #     lr = lr / 10
    # if step >= int(80000 * x):
    #     lr = lr / 10
    # opt = torch.optim.SGD(
    #     [{'params': weight_p, 'weight_decay': config.weight_decay, 'lr': lr},
    #      {'params': bias_p, 'lr': lr * config.bias_lr_factor}],
    #     momentum=0.9, )

    opt = torch.optim.Adam(
        [{'params': weight_p, 'weight_decay': config.weight_decay, 'lr': lr},
         {'params': bias_p, 'lr': lr * config.bias_lr_factor}],
    )
    dataset = Read_Data(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size_per_GPU, collate_fn=func,
                            shuffle=True, drop_last=True, pin_memory=True, num_workers=16)

    epochs = 10000
    flag = False
    print('start:  step=', step)
    for epoch in range(epochs):
        for imgs, bboxes, num_b, num_H, num_W, anchors, num_A in dataloader:

            loss = list(map(P_train.get_train_data, imgs, bboxes, num_b, num_H, num_W, anchors, num_A))
            loss = list(zip(*loss))
            logits = torch.cat(loss[0], dim=0)
            label = torch.cat(loss[1], dim=0)
            loc_train = torch.cat(loss[2], dim=0)
            loc = torch.cat(loss[3], dim=0)

            logits, label = get_cls_train_data(logits, label)

            cls_loss = get_cls_loss(logits, label, config.train_batch)
            bbox_loss = get_bbox_loss(loc_train, loc, int(config.train_batch / 2))
            loss = cls_loss + bbox_loss / 2

            opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(train_params, 10, norm_type=2)
            opt.step()

            if step % 20 == 0:
                print(datetime.now(), 'loss:%.4f' % loss, 'cls_loss:%.4f' % cls_loss,
                      'box_loss:%.4f' % bbox_loss, 'norm:%.4f' % norm, step, label.shape[0], '%d' % label.sum().data,
                      loc.shape[0], opt.param_groups[0]['lr'])
            step += 1

            # if step == int(60000 * x) or step == int(80000 * x):
            #     for param_group in opt.param_groups:
            #         param_group['lr'] = param_group['lr'] / 10
            #         print('*******************************************', param_group['lr'])

            if (step <= 10000 and step % 1000 == 0) or step % 5000 == 0 or step == 1:
                torch.save(model.state_dict(), './models/R_net_%d_1.pth' % step)
            if step >= 90010 * x:
                flag = True
                break
        if flag:
            break
    torch.save(model.state_dict(), './models/R_net_final_1.pth')

    pass


if __name__ == "__main__":
    path = '/home/zhai/PycharmProjects/Demo35/MTCNN/data_process_3/'
    files = [path + 'R_net_train_wider_1.pkl']
    config = Config(files, lr=0.001, weight_decay=0.0000, scale=0.79,
                    batch_size_per_GPU=14, gpus=1,
                    P_net_conf_thresh=0.5,
                    P_net_iou_thresh=0.7,
                    R_net_conf_thresh=0.5,
                    R_net_iou_thresh=0.7,
                    O_net_conf_thresh=0.5,
                    O_net_iou_thresh=0.7,
                    )
    step = 0
    x = 0.5
    model_file = ''
    R_train = R_net_train()
    train(R_train, config, step, x, model_file=model_file)

    pass
