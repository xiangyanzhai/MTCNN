# !/usr/bin/python
# -*- coding:utf-8 -*-
class Config():
    def __init__(self, files, lr=0.005,
                 weight_decay=0.0001,
                 scale=0.79,
                 batch_size_per_GPU=64, gpus=1,
                 train_batch=512,
                 P_net_conf_thresh=0.5,
                 P_net_iou_thresh=0.7,
                 R_net_conf_thresh=0.5,
                 R_net_iou_thresh=0.7,
                 O_net_conf_thresh=0.5,
                 O_net_iou_thresh=0.7,
                 roi_min_size=[6, 12, 12],
                 bias_lr_factor=1

                 ):
        self.files = files
        self.lr = lr
        self.weight_decay = weight_decay
        self.scale = scale

        self.batch_size_per_GPU = batch_size_per_GPU
        self.gpus = gpus
        self.train_batch = train_batch
        self.P_net_conf_thresh = P_net_conf_thresh
        self.P_net_iou_thresh = P_net_iou_thresh
        self.R_net_conf_thresh = R_net_conf_thresh
        self.R_net_iou_thresh = R_net_iou_thresh
        self.O_net_conf_thresh = O_net_conf_thresh
        self.O_net_iou_thresh = O_net_iou_thresh
        self.roi_min_size = roi_min_size
        self.bias_lr_factor = bias_lr_factor

        print('==============================================================')
        print('files:\t', self.files)
        print('lr:\t', self.lr)
        print('weight_decay:\t', self.weight_decay)
        print('scale:\t', self.scale)

        print('batch_size_per_GPU:\t', self.batch_size_per_GPU)
        print('gpus:\t', self.gpus)
        print('train_batch:\t', self.train_batch)
        print('==============================================================')
        print('P_net_conf_thresh :\t', self.P_net_conf_thresh)
        print('P_net_iou_thresh :\t', self.P_net_iou_thresh)
        print('==============================================================')
        print('R_net_conf_thresh :\t', self.R_net_conf_thresh)
        print('R_net_iou_thresh :\t', self.R_net_iou_thresh)
        print('==============================================================')
        print('O_net_conf_thresh :\t', self.O_net_conf_thresh)
        print('O_net_iou_thresh :\t', self.O_net_iou_thresh)
        print('==============================================================')
        print('roi_min_size :\t', self.roi_min_size)
        print('==============================================================')
        print('bias_lr_factor  :\t', self.bias_lr_factor)
        print('==============================================================')
