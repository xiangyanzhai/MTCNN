# !/usr/bin/python
# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import codecs
from sklearn.externals import joblib

path = '/home/zhai/PycharmProjects/Demo35/dataset/FDDB/'
if __name__ == "__main__":
    files = os.listdir('/home/zhai/PycharmProjects/Demo35/dataset/FDDB/FDDB-folds')
    files = sorted(files)
    R = {}
    i = 0
    for z in range(0, 20, 2):
        file = '/home/zhai/PycharmProjects/Demo35/dataset/FDDB/FDDB-folds/' + files[z]
        with codecs.open(file, 'r') as f:
            s = f.read().strip()
            s = s.split('\n')
            i = 0
            while i < len(s):
                name = s[i]
                num = int(s[i + 1])
                bboxes = s[i + 2:i + 2 + num]
                bboxes = [bbox.split() for bbox in bboxes]
                bboxes = np.array(bboxes)
                bboxes = bboxes.astype(np.float32)

                i = i + 2 + num
                R[name] = bboxes
    joblib.dump(R, 'ellips.pkl')
