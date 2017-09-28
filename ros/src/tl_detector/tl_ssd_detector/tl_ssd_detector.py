#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# Implementation of SSD and this code is taken from
# https://github.com/balancap/SSD-Tensorflow
#

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

class TLSSDDetector(object):

    def __init__(self):

        # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

        # Input placeholder.
        self.net_shape = (512, 512)
        data_format = 'NHWC'
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, self.net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(image_pre, 0)

        # Define the SSD model.
        reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_512.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            self.predictions, self.localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=reuse)

        # Restore SSD model.
        ckpt_filename = 'tl_ssd_detector/checkpoint/model.ckpt-226812'
        # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
        self.isess = tf.Session(config=config)
        self.isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.isess, ckpt_filename)

        # SSD default anchor boxes.
        self.ssd_anchors = ssd_net.anchors(self.net_shape)

    # Main image processing routine.
    def process_image(self, img, select_threshold=0.5, nms_threshold=.45):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=select_threshold, img_shape=self.net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes

    def draw_bboxes(self, img, select_threshold=0.5, nms_threshold=.45):
        classes, scores, bboxes = self.process_image(img, select_threshold, nms_threshold)
        colors = {}
        colors[1] = (
            0,
            255,
            255
        )
        visualization.bboxes_draw_on_img(img, classes, scores, bboxes, colors)

    # # Test on some demo image and visualize output.
    # # path = '../test/bosch-train/'
    # path = '../test/sim/'
    # image_names = sorted(os.listdir(path))
    # print(image_names)
    #
    # for image_name in image_names:
    #     if image_name != '.DS_Store':
    #         img = 255 * mpimg.imread(path + image_name)[:,:,:3]
    #         img = img.astype(np.uint8)
    #         print(np.min(img))
    #         print(np.max(img))
    #         rclasses, rscores, rbboxes =  process_image(img)
    #
    #         # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    #         visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
