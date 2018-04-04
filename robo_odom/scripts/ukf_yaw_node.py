#!/usr/bin/python2.7
# coding=<encoding name> 例如，可添加# coding=utf-8

###############UKF参数选取和系统方程测定方法##############
#参数选取原则  
#首先按照动力学方程建立F方程  这里我使用的是匀速运动方程，考虑到引入加速度可能会造成不稳定，而且匀速方程表现较好，因此使用匀速方程。 S（i+1） = S（i） + V*dt
#F要和H相乘，H=【yaw，yaw，dyaw】，所以写为下面的形式，两个yaw分别是imu和uwb的观测
#用H来选择有几个输入参数，这里我有三个参数，分别是yaw观测1，yaw观测2，dyaw。  
#dim_x = 2， 表示输入方程有三个输入  
#dim_z = 2 表示观测方程z有3个观测。yaw，dyaw 
#p_std_yaw,v_std_yaw 表示测量误差，这里根据测量结果填写,其中p表示融合后的yaw输出，v表示dyaw输出
#Q噪声项，var这里我发现用0.2的结果比较好，目前还不知道为什么  
#P里头分别填上前面yaw，yaw，dyaw能达到的最大值。  
#ukf.x = np.array([0., 0.])表示起始位置和起始速度都是0
# MerweScaledSigmaPoints(2, alpha=.1, beta=2., kappa=-1.0) 2表示输入参数有三个yaw，dyaw  
#######################################################
#数据融合方法说明：
#这里使用的数据融合方法是这样的：
#首先uwb更新数据（较低频率）30HZ
#然后在两个uwb的数据帧中间imu会以较高的频率更新数据100HZ
#这样就可以用uwb的位置作为初始位置，中间用imu作积分得到位置累加上去。
#最后融合的数据就能达到100HZ

#注意事项：开车前保持车辆静止

#################比赛场地坐标系定义###########################
#                  X+
#                  1
#                  0
#                  0
#                  0
#                  0
#                  0
#                  0
# Y+  <-000000000000

# Check list
# KALMAN_GAIN 取值
# imu yaw输出是否和uwb对应，不对应加上偏差值后是否正确



import rospy 
import roslib
import pickle
import math
from numpy.random import randn
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped

KALMAN_GAIN = 0.8
FUSE_IMUYAW = False

UWB_INIT_COUNTER = 0
YAW_BAIS = 0  #BAIS use for adjustment
IMU_INIT = True
UWB_INIT = False
THRESH = 1

vel_wheel_x = vel_wheel_y = 0
pos_uwb_x = pos_uwb_y = uwb_yaw = uwb_seq = 0
imu_yaw = 0
pre_yaw = yaw_counter = 0

ukf_result = []
uwb_yaw_list = []
imu_yaw_list = []

# S（i+1） = S（i） + V*dt
def f_cv(x, dt):
    """ state transition function for a 
    constant velocity aircraft"""

    F = np.array([[1, dt],
                  [0,  1]],dtype=float)
    return np.dot(F, x)


def h_cv(x):
    return np.array([x[0], x[1]])


def UKFinit():
    global ukf
    ukf_fuse = []
    p_std_yaw = 0.004
    v_std_yaw = 0.008
    dt = 0.0125 #80HZ


    sigmas = MerweScaledSigmaPoints(2, alpha=.1, beta=2., kappa=-1.0)
    ukf = UKF(dim_x=2, dim_z=2, fx=f_cv, hx=h_cv, dt=dt, points=sigmas)
    ukf.x = np.array([0., 0.,])
    ukf.R = np.diag([p_std_yaw, v_std_yaw]) 
    ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.2)
    ukf.P = np.diag([6.3, 1])

#算法简介：使用uwb的帧为基准帧，使用FUSE_IMUYAW控制是否按照比例融合IMU的yaw输出
#如果uwb一直不发生更新，那么就直接由imu的速度进行积分补偿，当接收到uwb帧时候就将uwb设置成基准帧。

def callback_imu(imu):
    global imu_last_time, IMU_INIT, vel_imu_yaw, imu_yaw, fuse_yaw, YAW_BAIS, FUSE_IMUYAW, KALMAN_GAIN
    global ukf_result, ukf, uwb_yaw, uwb_seq, last_uwb_seq, incremental_yaw, yaw_counter, pre_yaw, THRESH
    
    if IMU_INIT == True:
        print "ukf yaw Init Finished!"
        last_uwb_seq = 0
        incremental_yaw = 0
        imu_last_time = imu.header.stamp.secs + imu.header.stamp.nsecs * 10**-9
        pre_yaw = uwb_yaw
        IMU_INIT = False
    else:
        #print "ukf yaw Processing"
        imu_time = imu.header.stamp.secs + imu.header.stamp.nsecs * 10**-9
        dt = imu_time - imu_last_time

        vel_imu_yaw = imu.angular_velocity.z
        #TO much diveration in imu yaw, ignore for now.
        #qn_imu = [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]
        #(imu_roll,imu_pitch,imu_yaw) = euler_from_quaternion(qn_imu)
        #imu_yaw = imu_yaw + YAW_BAIS

        if uwb_yaw > -np.pi and uwb_yaw < -np.pi + THRESH and pre_yaw < np.pi and pre_yaw > np.pi - THRESH:
            yaw_counter = yaw_counter + 1
        if uwb_yaw < np.pi and uwb_yaw > np.pi - THRESH and pre_yaw > - np.pi and pre_yaw < - np.pi + THRESH:
            yaw_counter = yaw_counter - 1
        pre_yaw = uwb_yaw
        yaw_out = yaw_counter * 2 * np.pi + uwb_yaw
        
        
        # uwb not change, use imu to compensate
        if uwb_seq - last_uwb_seq == 0:
            incremental_yaw = incremental_yaw + vel_imu_yaw * dt
            fuse_yaw = yaw_out + incremental_yaw
        else:
            dt = imu_time - uwb_time
            incremental_yaw = 0
            incremental_yaw = incremental_yaw + vel_imu_yaw * dt
            fuse_yaw = yaw_out + incremental_yaw

        if FUSE_IMUYAW == True:
            fuse_yaw = KALMAN_GAIN * fuse_yaw + (1-KALMAN_GAIN) * imu_yaw
            

        imu_last_time = imu_time
        last_uwb_seq = uwb_seq

        ukf_input = [fuse_yaw, vel_imu_yaw]
        ukf.predict()
        ukf.update(ukf_input)
        ukf_out_yaw = ukf.x[0]
        
        ukf_out_yaw = ukf_out_yaw - yaw_counter * 2 * np.pi
        #print ukf.x[0]
        #print 'imu_yaw:',imu_yaw, 'uwb_yaw',uwb_yaw,'fuse_yaw', fuse_yaw,'bais:', uwb_yaw - imu_yaw, 'kalman',ukf_out_yaw, 'accz', vel_imu_yaw

        ukf_yaw = Odometry()
        ukf_yaw.header.frame_id = "ukf_yaw"
        ukf_yaw.header.stamp.secs = imu.header.stamp.secs
        ukf_yaw.header.stamp.nsecs = imu.header.stamp.nsecs

        (ukf_yaw.pose.pose.orientation.x, ukf_yaw.pose.pose.orientation.y, ukf_yaw.pose.pose.orientation.z, ukf_yaw.pose.pose.orientation.w) = quaternion_from_euler(0,0,ukf_out_yaw)

        pub_ukf_yaw.publish(ukf_yaw)




def callback_uwb(uwb):
    global uwb_yaw, uwb_seq, uwb_time, imu_yaw_list, uwb_yaw_list, UWB_INIT_COUNTER, YAW_BAIS, UWB_INIT



    uwb_seq = uwb.header.seq
    uwb_time = uwb.header.stamp.secs + uwb.header.stamp.nsecs * 10**-9

    qn_uwb = [uwb.pose.pose.orientation.x, uwb.pose.pose.orientation.y, uwb.pose.pose.orientation.z, uwb.pose.pose.orientation.w]
    (uwb_roll,uwb_pitch,uwb_yaw) = euler_from_quaternion(qn_uwb)

    if UWB_INIT == True:
        if imu_yaw != 0:
            print 'tik tiok'
            imu_yaw_list.append(imu_yaw)
            uwb_yaw_list.append(uwb_yaw)
            if UWB_INIT_COUNTER == 50:
                YAW_BAIS = np.sum(uwb_yaw_list)/50 - np.sum(imu_yaw_list)/50
                UWB_INIT = False
                print "UWB inis finished. YAW_BAIS = ", YAW_BAIS
            UWB_INIT_COUNTER = UWB_INIT_COUNTER + 1



rospy.init_node('ukf_yaw_node')
UKFinit()

subimu = rospy.Subscriber('/px4/imu/data', Imu, callback_imu)
subuwb = rospy.Subscriber('/map/uwb/data', Odometry, callback_uwb)

pub_ukf_yaw = rospy.Publisher('/ukf/yaw', Odometry, queue_size=1)
rospy.spin()