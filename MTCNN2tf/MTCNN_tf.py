# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import codecs
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.externals import joblib
from datetime import datetime
from MTCNN.tool.nms_tf import py_nms, nms
from MTCNN.tool.config import Config
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


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
    return tf.constant(anchors)


def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = -alphas * tf.nn.relu(-inputs)
    return pos + neg


def loc2bbox(pre_loc, anchor):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    hw = pre_loc[..., 2:4] * c_hw
    yx1 = anchor[..., :2] + pre_loc[..., :2] * c_hw
    yx2 = yx1 + hw
    bboxes = tf.concat((yx1, yx2), axis=-1)
    return bboxes


def bbox2square(bboxes):
    wh = bboxes[..., 2:4] - bboxes[..., :2]
    wh = tf.reduce_max(wh, axis=-1)
    wh = wh[:, None]
    xy = (bboxes[..., 2:4] + bboxes[..., :2]) / 2
    xy1 = xy - wh / 2
    xy1 = tf.maximum(xy1, 0)
    xy2 = xy1 + wh
    bboxes = tf.concat([xy1, xy2], axis=-1)
    return bboxes


class MTCNN():
    def __init__(self, config):
        self.anchors = get_anchors(5000)
        self.scale = config.scale
        self.config = config

    def P_net(self, net):
        with tf.variable_scope('P_net'):
            with slim.arg_scope([slim.conv2d], activation_fn=prelu, padding='VALID'):
                net = slim.conv2d(net, 10, [3, 3], scope='conv1', )
                net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool1', padding='SAME')
                net = slim.conv2d(net, num_outputs=16, kernel_size=[3, 3], stride=1, scope='conv2')
                net = slim.conv2d(net, num_outputs=32, kernel_size=[3, 3], stride=1, scope='conv3', )
                net_logits = slim.conv2d(net, num_outputs=2, kernel_size=[1, 1], stride=1, scope='conv_cls',
                                         activation_fn=None)
                net_loc = slim.conv2d(net, num_outputs=4, kernel_size=[1, 1], stride=1, scope='conv_offset',
                                      activation_fn=None)
                net_landmarks = slim.conv2d(net, num_outputs=10, kernel_size=[1, 1], stride=1, scope='conv_landmarks',
                                            activation_fn=None)

        return net_logits, net_loc, net_landmarks

    def R_net(self, net):
        with tf.variable_scope('R_net'):
            with slim.arg_scope([slim.conv2d], activation_fn=prelu, padding='VALID'):
                net = slim.conv2d(net, num_outputs=28, kernel_size=[3, 3], stride=1, scope="conv1")
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
                net = slim.conv2d(net, num_outputs=48, kernel_size=[3, 3], stride=1, scope="conv2")
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
                net = slim.conv2d(net, num_outputs=64, kernel_size=[2, 2], stride=1, scope="conv3")
                net = tf.transpose(net, (0, 3, 1, 2))
                net = tf.reshape(net, (-1, 576))

                fc1 = slim.fully_connected(net, num_outputs=128, scope="fc1", activation_fn=prelu)

                net_logits = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc", activation_fn=None)

                net_loc = slim.fully_connected(fc1, num_outputs=4, scope="bbox_fc", activation_fn=None)

                net_landmarks = slim.fully_connected(fc1, num_outputs=10, scope="landmark_fc", activation_fn=None)
        return net_logits, net_loc, net_landmarks

    def O_net(self, net):
        with tf.variable_scope('O_net'):
            with slim.arg_scope([slim.conv2d], activation_fn=prelu, padding='VALID'):
                net = slim.conv2d(net, num_outputs=32, kernel_size=[3, 3], stride=1, scope="conv1")
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
                net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=1, scope="conv2")
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
                net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=1, scope="conv3")
                net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
                net = slim.conv2d(net, num_outputs=128, kernel_size=[2, 2], stride=1, scope="conv4")
                net = tf.transpose(net, (0, 3, 1, 2))
                net = tf.reshape(net, (-1, 1152))
                fc1 = slim.fully_connected(net, num_outputs=256, scope="fc1", activation_fn=prelu)

                net_logits = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc", activation_fn=None)

                net_loc = slim.fully_connected(fc1, num_outputs=4, scope="bbox_fc", activation_fn=None)

                net_landmarks = slim.fully_connected(fc1, num_outputs=10, scope="landmark_fc", activation_fn=None)
        return net_logits, net_loc, net_landmarks

    def body(self, i, P_out):

        h = self.H * self.scale ** i
        w = self.W * self.scale ** i
        im = tf.image.resize_images(self.input, size=(tf.to_int32(h), tf.to_int32(w)))

        n_h, n_w = tf.to_int32(tf.ceil(h / 12) * 12), tf.to_int32(tf.ceil(w / 12) * 12)

        im = tf.pad(im, [[0, 0], [0, n_h - tf.to_int32(h)], [0, n_w - tf.to_int32(w)], [0, 0]], mode="CONSTANT",
                    constant_values=127.5)
        im.set_shape(tf.TensorShape([None, None, None, 3]))

        im = (im - 127.5) / 128.0
        net_logits, net_loc, net_landmarks = self.P_net(im)

        map_H, map_W = tf.shape(net_logits)[1], tf.shape(net_logits)[2]

        tanchors = tf.reshape(self.anchors[:map_H, :map_W], (-1, 4)) / self.scale ** i
        net_logits = tf.reshape(net_logits, (-1, 2))
        net_loc = tf.reshape(net_loc, (-1, 4))
        net_landmarks = tf.reshape(net_landmarks, (-1, 10))

        r = tf.concat([net_logits, net_loc, net_landmarks, tanchors], axis=-1)

        score = tf.nn.softmax(r[:, :2])[:, 1]
        inds = score >= self.config.P_net_conf_thresh
        r = tf.boolean_mask(r, inds)
        bboxes = loc2bbox(r[:, 2:6], r[:, -4:])
        bboxes = tf.clip_by_value(bboxes, [0, 0, 0, 0], [self.W - 1, self.H - 1, self.W - 1, self.H - 1])
        hw = bboxes[:, 2:4] - bboxes[:, :2]
        inds = tf.reduce_all(hw >= self.config.roi_min_size[0], axis=-1)
        inds = tf.reshape(inds, (-1,))
        bboxes = tf.boolean_mask(bboxes, inds)
        score = tf.boolean_mask(score, inds)

        score, top_k = tf.nn.top_k(score, k=tf.shape(score)[0])
        bboxes = tf.gather(bboxes, top_k)
        keep = tf.image.non_max_suppression(bboxes, score, tf.shape(bboxes)[0],
                                            iou_threshold=0.5)
        bboxes = tf.gather(bboxes, keep)
        score = tf.gather(score, keep)

        score = tf.reshape(score, (-1, 1))

        r = tf.concat([bboxes, score], axis=-1)

        P_out = tf.concat((P_out, r), axis=0)

        i += 1
        return i, P_out

    def handle_P_out(self, P_out, h, w):

        bboxes = P_out[:, :4]
        score = P_out[:, 4]

        score, top_k = tf.nn.top_k(score, k=tf.shape(score)[0])
        bboxes = tf.gather(bboxes, top_k)

        keep = tf.image.non_max_suppression(bboxes, score, tf.shape(bboxes)[0],
                                            iou_threshold=self.config.P_net_iou_thresh)
        bboxes = tf.gather(bboxes, keep)
        score = tf.gather(score, keep)

        score = tf.reshape(score, (-1, 1))

        return tf.concat([bboxes, score], axis=-1)

    def get_R_net_res(self, tanchors, h, w, im):
        im = tf.to_float(im)
        roi = bbox2square(tanchors[:, :4])
        n_w, n_h = tf.reduce_max(tanchors[:, 2]), tf.reduce_max(tanchors[:, 3])
        n_w, n_h = tf.to_int32(tf.ceil(n_w)), tf.to_int32(tf.ceil(n_h))
        im = tf.pad(im, [[0, 0], [0, tf.reduce_max([n_h - tf.to_int32(h), 0])],
                         [0, tf.reduce_max([n_w - tf.to_int32(w), 0])], [0, 0]], mode="CONSTANT",
                    constant_values=127.5)
        im.set_shape(tf.TensorShape([None, None, None, 3]))
        roi_norm = roi / tf.concat([[w], [h], [w], [h]], axis=0)

        roi_norm = tf.gather(roi_norm, [1, 0, 3, 2], axis=-1)

        im = tf.image.crop_and_resize(im, roi_norm, tf.zeros(tf.shape(roi)[0], dtype=tf.int32), (24, 24))
        im = (im - 127.5) / 128.0
        net_logits, net_loc, net_landmarks = self.R_net(im)

        score = tf.nn.softmax(net_logits, axis=-1)[..., 1]

        inds = score > self.config.R_net_conf_thresh

        score = tf.boolean_mask(score, inds)
        net_loc = tf.boolean_mask(net_loc, inds)
        roi = tf.boolean_mask(roi, inds)
        bboxes = loc2bbox(net_loc, roi)

        bboxes = tf.clip_by_value(bboxes, [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
        hw = bboxes[:, 2:4] - bboxes[:, :2]
        inds = tf.reduce_all(hw >= self.config.roi_min_size[1], axis=-1)

        inds = tf.reshape(inds, (-1,))
        bboxes = tf.boolean_mask(bboxes, inds)
        score = tf.boolean_mask(score, inds)

        score, top_k = tf.nn.top_k(score, k=tf.shape(score)[0])
        bboxes = tf.gather(bboxes, top_k)

        keep = tf.image.non_max_suppression(bboxes, score, tf.shape(bboxes)[0],
                                            iou_threshold=self.config.R_net_iou_thresh)
        bboxes = tf.gather(bboxes, keep)
        score = tf.gather(score, keep)

        score = tf.reshape(score, (-1, 1))

        return tf.concat([bboxes, score], axis=-1)

    def get_O_net_res(self, tanchors, h, w, im):
        im = tf.to_float(im)
        roi = bbox2square(tanchors[:, :4])
        n_w, n_h = tf.reduce_max(tanchors[:, 2]), tf.reduce_max(tanchors[:, 3])
        n_w, n_h = tf.to_int32(tf.ceil(n_w)), tf.to_int32(tf.ceil(n_h))
        im = tf.pad(im, [[0, 0], [0, tf.reduce_max([n_h - tf.to_int32(h), 0])],
                         [0, tf.reduce_max([n_w - tf.to_int32(w), 0])], [0, 0]], mode="CONSTANT",
                    constant_values=127.5)
        im.set_shape(tf.TensorShape([None, None, None, 3]))
        roi_norm = roi / tf.concat([[w], [h], [w], [h]], axis=0)

        roi_norm = tf.gather(roi_norm, [1, 0, 3, 2], axis=-1)

        im = tf.image.crop_and_resize(im, roi_norm, tf.zeros(tf.shape(roi)[0], dtype=tf.int32), (48, 48))
        im = (im - 127.5) / 128.0
        net_logits, net_loc, net_landmarks = self.O_net(im)

        score = tf.nn.softmax(net_logits, axis=-1)[..., 1]

        inds = score > self.config.O_net_conf_thresh

        score = tf.boolean_mask(score, inds)
        net_loc = tf.boolean_mask(net_loc, inds)
        roi = tf.boolean_mask(roi, inds)
        bboxes = loc2bbox(net_loc, roi)

        bboxes = tf.clip_by_value(bboxes, [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
        hw = bboxes[:, 2:4] - bboxes[:, :2]
        inds = tf.reduce_all(hw >= self.config.roi_min_size[2], axis=-1)

        inds = tf.reshape(inds, (-1,))
        bboxes = tf.boolean_mask(bboxes, inds)
        score = tf.boolean_mask(score, inds)

        score, top_k = tf.nn.top_k(score, k=tf.shape(score)[0])
        bboxes = tf.gather(bboxes, top_k)

        keep = nms(bboxes, self.config.O_net_iou_thresh)
        bboxes = tf.gather(bboxes, keep)
        score = tf.gather(score, keep)

        score = tf.reshape(score, (-1, 1))

        return tf.concat([bboxes, score], axis=-1)

    def build_net(self):
        self.input = tf.placeholder(dtype=tf.uint8, shape=(None, None, None, 3), name='input')
        im = self.input
        H = tf.shape(im)[1]
        W = tf.shape(im)[2]

        H = tf.to_float(H)
        W = tf.to_float(W)
        self.H = H
        self.W = W

        i = 0.0
        P_out = tf.zeros([0, 5], dtype=tf.float32)
        _, P_out = tf.while_loop(lambda i, _: tf.reduce_all([H * self.scale ** i >= 12, W * self.scale ** i >= 12]),
                                 self.body, [i, P_out],
                                 shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, 5])], back_prop=False)

        tanchors = self.handle_P_out(P_out, H, W)[:50000]
        tanchors = self.get_R_net_res(tanchors, H, W, self.input)
        tanchors = self.get_O_net_res(tanchors, H, W, self.input)
        tanchors = tf.identity(tanchors, name='out')
        self.r = tanchors

    def save_pb(self):

        self.build_net()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        P_vars = [v for v in tf.global_variables() if 'P_net' in v.name]
        R_vars = [v for v in tf.global_variables() if 'R_net' in v.name]
        O_vars = [v for v in tf.global_variables() if 'O_net' in v.name]
        P = joblib.load('./models/P.pkl')
        R = joblib.load('./models/R.pkl')
        O = joblib.load('./models/O.pkl')
        print(len(P), len(P_vars), len(R), len(R_vars))
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(len(P_vars)):
                print(P_vars[i].name, P_vars[i])
                sess.run(P_vars[i].assign(P[i]))
            print('*****************************************')
            for i in range(len(R_vars)):
                print(R_vars[i].name, R_vars[i], R[i].shape)
                sess.run(R_vars[i].assign(R[i]))
            print('*****************************************')
            for i in range(len(O_vars)):
                print(O_vars[i].name, O_vars[i], O[i].shape)
                sess.run(O_vars[i].assign(O[i]))
            print('*****************************************')

            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['out'])
            with tf.gfile.FastGFile('model.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

    def test_pb(self):
        pb_file = 'model.pb'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session()
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
            constant_values = {}
            constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
            for constant_op in constant_ops:
                print(constant_op.name, )
            # 需要有一个初始化的过程
            #     sess.run(tf.global_variables_initializer())
            input_x = sess.graph.get_tensor_by_name('input:0')
            out = sess.graph.get_tensor_by_name('out:0')

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

                        res = sess.run(out, feed_dict={input_x: img[None]})

                        # draw_gt(img, res[..., :4])
                        m = res.shape[0]
                        print(datetime.now(), z, m)
                        f.write(file + '\n')
                        f.write(str(m) + '\n')
                        res[..., 2:4] = res[..., 2:4] - res[..., :2]
                        for bbox in res:
                            f.write(
                                str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ' + str(
                                    bbox[4]) + '\n')

    def test_pb_2(self):
        pb_file = 'model.pb'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session()
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
            constant_values = {}
            constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
            for constant_op in constant_ops:
                print(constant_op.name, )
            # 需要有一个初始化的过程
            #     sess.run(tf.global_variables_initializer())
            input_x = sess.graph.get_tensor_by_name('input:0')
            out = sess.graph.get_tensor_by_name('out:0')
            file = '/home/zhai/PycharmProjects/Demo35/MTCNN_tf/2.jpg'
            img = cv2.imread(file)
            res = sess.run(out, feed_dict={input_x: img[None]})
            inds = res[:, -1] > 0.8
            res = res[inds]
            draw_gt(img, res[..., :4])

    def test(self):
        self.build_net()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        P_vars = [v for v in tf.global_variables() if 'P_net' in v.name]
        R_vars = [v for v in tf.global_variables() if 'R_net' in v.name]
        O_vars = [v for v in tf.global_variables() if 'O_net' in v.name]
        P = joblib.load('./models/P.pkl')
        R = joblib.load('./models/R.pkl')
        O = joblib.load('./models/O.pkl')
        print(len(P), len(P_vars), len(R), len(R_vars))
        with tf.Session(config=config) as sess:
            # sess.run(tf.global_variables_initializer())

            for i in range(len(P_vars)):
                print(P_vars[i].name, P_vars[i])
                sess.run(P_vars[i].assign(P[i]))
            print('*****************************************')
            for i in range(len(R_vars)):
                print(R_vars[i].name, R_vars[i], R[i].shape)
                sess.run(R_vars[i].assign(R[i]))
            print('*****************************************')
            for i in range(len(O_vars)):
                print(O_vars[i].name, O_vars[i], O[i].shape)
                sess.run(O_vars[i].assign(O[i]))
            print('*****************************************')

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

                        res = sess.run(self.r, feed_dict={self.input: img[None]})

                        # draw_gt(img, res[..., :4])
                        m = res.shape[0]
                        print(datetime.now(), z, m)
                        f.write(file + '\n')
                        f.write(str(m) + '\n')
                        res[..., 2:4] = res[..., 2:4] - res[..., :2]
                        for bbox in res:
                            f.write(
                                str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ' + str(
                                    bbox[4]) + '\n')


def draw_gt(im, gt):
    im = im.astype(np.uint8).copy()
    boxes = gt.astype(np.int32)
    print(im.max(), im.min(), im.shape)
    for box in boxes:
        # print(box)
        x1, y1, x2, y2 = box[:4]

        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))

    im = im.astype(np.uint8)
    cv2.imshow('a', im)
    cv2.waitKey(20000)
    return im


if __name__ == "__main__":
    config = Config(None, scale=0.79,
                    P_net_conf_thresh=0.5,
                    P_net_iou_thresh=0.7,
                    R_net_conf_thresh=0.5,
                    R_net_iou_thresh=0.7,
                    O_net_conf_thresh=0.5,
                    O_net_iou_thresh=0.7, )
    mtcnn = MTCNN(config)
    # mtcnn.test()
    # mtcnn.save_pb()
    mtcnn.test_pb()
    # mtcnn.test_pb_2()
