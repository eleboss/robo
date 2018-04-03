#!/usr/bin/python2.7
# coding=<encoding name> 例如，可添加# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rospy 
import roslib
import time
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import cv2
import sys
import os
import glob

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

import tensorflow as tf
from config import *
from train import _draw_box
from nets import *
count = 0
flag = True

def DetectInit():
    global sess, model, mc

    detect_net = 'squeezeDet'
    checkpoint = '/home/ubuntu/data/model_checkpoints/squeezeDet/model.ckpt-26000'


    assert detect_net == 'squeezeDet' or detect_net == 'squeezeDet+', \
        'Selected nueral net architecture not supported'

    tf.Graph().as_default()
    # Load model
    if detect_net == 'squeezeDet':
        mc = kitti_squeezeDet_config()
        mc.BATCH_SIZE = 1
        # model parameters will be restored from checkpoint
        mc.LOAD_PRETRAINED_MODEL = False
        model = SqueezeDet(mc, '0')
    elif detect_net == 'squeezeDet+':
        mc = kitti_squeezeDetPlus_config()
        mc.BATCH_SIZE = 1
        mc.LOAD_PRETRAINED_MODEL = False
        model = SqueezeDetPlus(mc, '0')

    saver = tf.train.Saver(model.model_params)
    # Use jit xla
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #with tf.Session(config=config) as sess:
    sess = tf.Session(config=config) 
    saver.restore(sess, checkpoint)

def TsDet_callback(rgb,depth):
    global count, sess, model, mc
    print ('I here rgb and depth !',count)
    count = count + 1

    bridge = CvBridge()
    try:
        cv_image_rgb = bridge.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
        cv_image_depth = bridge.imgmsg_to_cv2(depth, desired_encoding="8UC1")
    except CvBridgeError as e:
        print(e)

    #position_pub.publish(rgb)
    #visualize
    #cv2.imshow("Image window", cv_image_depth)
    #cv2.waitKey(3)

    times = {}
    t_start = time.time()


    im = cv_image_rgb
    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
    input_image = im - mc.BGR_MEANS

    t_reshape = time.time()
    times['reshape']= t_reshape - t_start

    # Detect
    det_boxes, det_probs, det_class = sess.run(
        [model.det_boxes, model.det_probs, model.det_class],
        feed_dict={model.image_input:[input_image]})

    t_detect = time.time()
    times['detect']= t_detect - t_reshape

    # Filter
    final_boxes, final_probs, final_class = model.filter_prediction(
        det_boxes[0], det_probs[0], det_class[0])

    # keep_idx    = [idx for idx in range(len(final_probs)) \
    #                   if final_probs[idx] > mc.PLOT_PROB_THRESH]
    # final_boxes = [final_boxes[idx] for idx in keep_idx]
    # final_probs = [final_probs[idx] for idx in keep_idx]
    # final_class = [final_class[idx] for idx in keep_idx]
    keep_idx = np.squeeze(np.argwhere(np.array(final_probs) > mc.PLOT_PROB_THRESH))

    final_boxes = np.array(final_boxes)[keep_idx, :]
    final_probs = np.array(final_probs)[keep_idx]
    final_class = np.array(final_class)[keep_idx]

    t_filter = time.time()
    times['filter']= t_filter - t_detect
    times['total']= time.time() - t_start
    time_str = 'Total time: {:.4f}, detection time: {:.4f}, filter time: '\
                '{:.4f}'. \
        format(times['total'], times['detect'], times['filter'])

    print (time_str)



    # # TODO(bichen): move this color dict to configuration file
    # cls2clr = {
    #     'red': (255, 191, 0),
    #     'wheel': (0, 191, 255),
    #     'blue':(128, 128, 0)
    # }

    # # Draw boxes
    # _draw_box(
    #     im, final_boxes,
    #     [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
    #         for idx, prob in zip(final_class, final_probs)],
    #     cdict=cls2clr,
    # )

    # file_name = os.path.split(f)[1]
    # out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
    # cv2.imwrite(out_file_name, im)
    # print ('Image detection output saved to {}'.format(out_file_name))




rospy.init_node('rgb_detection')
DetectInit()

rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
#rgb_sub = message_filters.Subscriber('/camera/infra1/image_rect_raw', Image)
depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_infra1/image_raw', Image)

#TODO 在后面的试验中调调整slop
TsDet = message_filters.ApproximateTimeSynchronizer([rgb_sub,depth_sub],queue_size=5, slop=0.1, allow_headerless=False)
TsDet.registerCallback(TsDet_callback)

#position_pub = rospy.Publisher('/test', Image, queue_size=1)
rospy.spin()