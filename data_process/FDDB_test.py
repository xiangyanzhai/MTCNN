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
    R=[]
    i = 0
    for z in range(1, 20, 2):
        file = '/home/zhai/PycharmProjects/Demo35/dataset/FDDB/FDDB-folds/' + files[z]
        with codecs.open(file, 'r') as f:
            s = f.read().strip()
            s = s.split('\n')
            R.append(s)
    joblib.dump(R, 'FDDB_test.pkl')
