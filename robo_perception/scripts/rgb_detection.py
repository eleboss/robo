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
import tf
import tf2_ros

from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry

import tensorflow as tf
from config import *
from train import _draw_box
from nets import *




count = 0
flag = True

def DetectInit():
    global sess, model, mc

    detect_net = 'squeezeDet'
    checkpoint = '/home/ubuntu/catkin_ws/src/robo_perception/scripts/weights/model.ckpt-99999'


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

def TsDet_callback(rgb,pointcloud):
    global count, sess, model, mc
    print ('I here rgb and pointcloud !',count)
    count = count + 1

    bridge = CvBridge()
    try:
        cv_image_rgb = bridge.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
        #cv_image_depth = bridge.imgmsg_to_cv2(depth, desired_encoding="16UC1")
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
    #print(final_boxes, final_probs, final_class)
    #print("keep", len(keep_idx))
    #keep_idx_ = keep_idx.tolist()
    #print(keep_idx)
    if keep_idx.shape == ():
        final_boxes = final_boxes[np.newaxis, :]
        final_probs = np.array([final_probs])
        final_class = np.array([final_class])
    #print(final_boxes, final_probs, final_class)
    avgX = 0
    avgY = 0
    avgZ = 0
    
    if len(final_boxes)>0:
        # pointcloud
        #print('find enemy')
        area = [bbox[2] * bbox[3] for bbox in final_boxes]
        max_area_idx = np.argmax(area)
        #print(final_boxes, max_area_idx)
        robo_bbox = final_boxes[max_area_idx, :]
        
        cx = robo_bbox[0]
        cy = robo_bbox[1]
        pointcloud_w = 5
        pointcloud_h = 5
        if pointcloud_w > robo_bbox[2]:
            pointcloud_w = robo_bbox[2]
        if pointcloud_h > robo_bbox[3]:
            pointcloud_h = robo_bbox[3]

        x_ = np.arange(int(cx - pointcloud_w/2), int(cx + pointcloud_w/2), 1)
        y_ = np.arange(int(cy - pointcloud_h/2), int(cy + pointcloud_h/2), 1)
        roi = [[x, y] for x in x_ for y in y_]

        points = list(pc2.read_points(pointcloud, skip_nans=False, field_names = ("x", "y", "z"), uvs=roi))
        # points_old = [[p[0],p[1],p[2]] for p in pc2.read_points(pointcloud, skip_nans=False, field_names=('x', 'y', 'z'))]
        # points_old = np.array(points_old)
        # points_old = np.resize(points_old, (240, 424 , 3))
        #print(points)
        robo_pointcloud = np.array(points)
        avgX = 0
        avgY = 0
        avgZ = 0
        idx = 0
        # robo_pointcloud = [if for pointcloud_ in robo_pointcloud]
        positionX = robo_pointcloud[:,0].reshape(-1,25).squeeze()
        positionY = robo_pointcloud[:,1].reshape(-1,25).squeeze()
        positionZ = robo_pointcloud[:,2].reshape(-1,25).squeeze()
        #print(positionX)
        
        positionX = positionX[np.logical_not(np.isnan(positionX))]
        positionY = positionY[np.logical_not(np.isnan(positionY))]
        positionZ = positionZ[np.logical_not(np.isnan(positionZ))]
        #print(positionX, positionY, positionZ)
        avgX = np.mean(positionX)
        avgY = np.mean(positionY)
        avgZ = np.mean(positionZ)
        print('position: ', avgX, avgY, avgZ)
        
        
        br = tf2_ros.TransformBroadcaster()
        t = TransformStamped()
        
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'enemy_0'
        t.transform.translation.x = avgZ
        t.transform.translation.y = -avgX
        t.transform.translation.z = 0
        t.transform.rotation.x = 0 
        t.transform.rotation.y = 0
        t.transform.rotation.z = 0
        t.transform.rotation.w = 1
        
        br.sendTransform(t) 

        enemy_position = Odometry()
        enemy_position.pose.pose.position.x = avgZ
        enemy_position.pose.pose.position.y = -avgX
        enemy_position.pose.pose.position.z = avgY
        pub.publish(enemy_position)
    else:
        print('No enemy!!!')	

    t_filter = time.time()
    times['filter']= t_filter - t_detect
    times['total']= time.time() - t_start
    time_str = 'Total time: {:.4f}, detection time: {:.4f}, filter time: '\
                '{:.4f}'. \
        format(times['total'], times['detect'], times['filter'])
    print (time_str)

    # # TODO(bichen): move this color dict to configuration file
    cls2clr = {
        'red': (0, 0, 255),
        'wheel': (0, 255, 0),
        'blue':(255, 0, 0)
    }

    # for idx, prob in zip(final_class, final_probs):
    #     print(mc.CLASS_NAMES[idx])
    #     print(idx)
    # Draw boxes
    if mc.DRAW_BOX and len(final_boxes):
        im = _draw_box(
            im, 
            final_boxes,
            [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(final_class, final_probs)],
            cdict=cls2clr,
        )
        # cv2.imshow('demo', im)
        # file_name = os.path.split(f)[1]
        file_name = '0001.jpg'
        out_file_name = os.path.join('/home/ubuntu/catkin_ws/src/robo_perception/scripts/visual', 'out_'+file_name)
        cv2.imwrite(out_file_name, im)
        print ('Image detection output saved to {}'.format(out_file_name))


rospy.init_node('rgb_detection')
DetectInit()

rgb_sub = message_filters.Subscriber('camera/color/image_raw', Image)
#rgb_sub = message_filters.Subscriber('camera/infra1/image_rect_raw', Image)
pc_sub =  message_filters.Subscriber('/camera/depth_registered/points', PointCloud2)
#depth_sub = message_filters.Subscriber('camera/depth/image_rect_raw', Image)
pub = rospy.Publisher('rgb_detection/enemy_position', Odometry, queue_size=1)

#TODO 在后面的试验中调调整slop
TsDet = message_filters.ApproximateTimeSynchronizer([rgb_sub,pc_sub],queue_size=5, slop=0.1, allow_headerless=False)
TsDet.registerCallback(TsDet_callback)

rospy.spin()