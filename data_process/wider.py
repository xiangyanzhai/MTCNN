# !/usr/bin/python
# -*- coding:utf-8 -*-
import codecs
import numpy as np
import cv2
from sklearn.externals import joblib


def handle(s):
    R = []
    R2 = []
    R3 = []
    for i in s:
        i = i.split()
        i = list(map(int, i))
        i = np.array(i)
        x1, y1, w, h = i[:4]
        x2, y2 = x1 + w, y1 + h
        if w * h == 0:
            continue
        if i[4] == 2:
            R3.append([x1, y1, x2, y2] + [0] * 10)
            continue
        if i[8] > 0:
            R2.append([x1, y1, x2, y2] + [0] * 10)
            continue
        if i[:4].sum() == 0:
            continue

        R.append([x1, y1, x2, y2] + [0] * 10)
    m = max(len(R), 400)
    R = R + R2 + R3
    R=R[:m]
    if R:
        R = np.array(R)
        R = R.reshape(-1, 14)
        R = R.astype(np.float32)
        return R
    # print('***********')
    return np.zeros((0, 14), dtype=np.float32)


img_Dir = '/home/zhai/PycharmProjects/Demo35/dataset/WIDER/WIDER_train/images/'
wider = []
if __name__ == "__main__":
    wider_face_file = '/home/zhai/PycharmProjects/Demo35/dataset/WIDER/wider_face_split/wider_face_train_bbx_gt.txt'
    with codecs.open(wider_face_file, 'r') as f:
        s = f.read().strip()
    s = s.split('\n')

    R = []
    i = 0
    c = 0
    ma = 0
    mi = 10000000000
    ma2 = 0
    while i < len(s):

        img_file = img_Dir + s[i]

        num = int(s[i + 1])
        if num == 0:
            num = 1
        bboxes = handle(s[i + 2:i + 2 + num])
        # bboxes=bboxes[:300]
        bboxes = bboxes - 1
        bboxes = np.maximum(bboxes, 0)
        ma = max(ma, bboxes.shape[0])
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        inds = areas > 72
        bboxes = bboxes[inds]

        if bboxes.shape[0] > 0:

            if areas.max() > 804 ** 2:
                print(bboxes.shape, areas.max(), areas.max() ** 0.5)
                pass

            ma2 = max(ma2, areas.max())
            mi = min(mi, areas.min())
            wider.append([img_file, bboxes])
        R.append([img_file, bboxes])
        i = i + 2 + num
    print(len(R))
    print(len(wider))
    print(ma)
    print(ma2, ma2 ** 0.5)
    print(mi)
    joblib.dump(wider, 'wider.pkl')
