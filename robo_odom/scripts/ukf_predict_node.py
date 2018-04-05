#!/usr/bin/python2.7
# coding=<encoding name> 例如，可添加# coding=utf-8

###############UKF参数选取和系统方程测定方法##############
#参数选取原则  
#首先按照动力学方程建立F方程 速度等于加速度乘时间。 v（i+1） = v（i） + a*dt
#F要和H相乘，H=【vx,vx',vy,vy'】，所以写为下面的形式，两个yaw分别是imu和uwb的观测
#用H来选择有几个输入参数，这里我有三个参数，分别是yaw观测1，yaw观测2，dyaw。  
#dim_x = 4， 表示输入方程有四个输入  
#dim_z = 4 表示观测方程z有4个观测。两个轴，xy，速度和加速度
#v_std_x, a_std_x, 表示速度和加速度的测量误差，这里根据测量结果填写
#Q噪声项，var这里我发现用0.2的结果比较好，目前还不知道为什么  
#P里头分别填上前面两轴速度加速度能达到的最大值。  【vx,vx',vy,vy'】
#ukf.x = np.array([0., 0., 0., 0.])表示起始位置和起始速度都是0
# MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1.0) 4表示输入参数有三个yaw，yaw，dyaw  
#######################################################
#数据融合方法说明：
#这里使用的数据融合方法是这样的：
#使用KALMAN_GAIN调整对应于轮式里程计和IMU的取值，一边取一点然后相加即可。


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
# KALMAN_GAIN是不是对应上去了

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


KALMAN_GAIN = 0.5 #use kalman gain to choce weather to fuse wheel velocity or not.
IMU_INIT = True
vel_wheel_x = vel_wheel_y = 0
pos_uwb_x = pos_uwb_y = uwb_yaw = uwb_seq = 0
ukf_yaw = 0

ukf_result = []

# S（i+1） = S（i） + V*dt
def f_cv(x, dt):
    """ state transition function for a 
    constant velocity aircraft"""
    
    F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]],dtype=float)
    return np.dot(F, x)

def h_cv(x):
    return np.array([x[0], x[1], x[2], x[3]])


def UKFinit():
    global ukf
    ukf_fuse = []
    v_std_x, v_std_y = 0.1, 0.1
    a_std_x, a_std_y = 0.1, 0.1
    dt = 0.0125 #80HZ


    sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1.0)
    ukf = UKF(dim_x=4, dim_z=4, fx=f_cv, hx=h_cv, dt=dt, points=sigmas)
    ukf.x = np.array([0., 0., 0., 0.])
    ukf.R = np.diag([v_std_x, a_std_x, v_std_y, a_std_y]) 
    ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.2)
    ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.2)
    ukf.P = np.diag([2, 1.5 ,2, 1.5])



def callback_imu(imu):
    global imu_last_time, IMU_INIT, acc_imu_x, acc_imu_y, vel_imu_x, vel_imu_y, vel_imu_yaw, imu_yaw
    global ukf_result, ukf, vel_wheel_x, vel_wheel_y, pos_uwb_x, pos_uwb_y, uwb_yaw, uwb_seq
    if IMU_INIT == True:
        print "ukf vel Init Finished!"
        imu_last_time = imu.header.stamp.secs + imu.header.stamp.nsecs * 10**-9
        vel_imu_x = 0
        vel_imu_y = 0
        IMU_INIT = False
    else:
        imu_time = imu.header.stamp.secs + imu.header.stamp.nsecs * 10**-9

        acc_imu_x = imu.linear_acceleration.x
        acc_imu_y = imu.linear_acceleration.y

        dt = imu_time - imu_last_time

        vel_imu_x = vel_imu_x + acc_imu_x * dt
        vel_imu_y = vel_imu_y + acc_imu_y * dt
        print 'imu acc', acc_imu_x

        imu_last_time = imu_time

        vel_fuse_x = KALMAN_GAIN * vel_wheel_x + (1-KALMAN_GAIN) * vel_imu_x
        vel_fuse_y = KALMAN_GAIN * vel_wheel_y + (1-KALMAN_GAIN) * vel_imu_y

        ukf_input = [vel_fuse_x, acc_imu_x, vel_fuse_y, acc_imu_y]
        ukf.predict()
        ukf.update(ukf_input)
        #print ukf.x
        ukf_out_vel_x = ukf.x[0]
        ukf_out_vel_y = ukf.x[2]

        ukf_vel = Odometry()
        ukf_vel.header.frame_id = "ukf_vel"
        ukf_vel.header.stamp.secs = imu.header.stamp.secs
        ukf_vel.header.stamp.nsecs = imu.header.stamp.nsecs
        ukf_vel.twist.twist.linear.x = ukf_out_vel_x
        ukf_vel.twist.twist.linear.y = ukf_out_vel_y
        pub_ukf_vel.publish(ukf_vel)



def callback_wheel(wheel):
    global vel_wheel_x, vel_wheel_y, ukf_yaw

    vel_wheel_x_ori = wheel.vector.x
    vel_wheel_y_ori = wheel.vector.y
    #if vel_wheel_x != 0 and vel_wheel_y!=0:
    #print 'before',vel_wheel_x_ori, vel_wheel_y_ori
    
    # project to the map link
    vel_wheel_x = vel_wheel_x_ori * np.cos(ukf_yaw) + vel_wheel_y_ori * np.sin(ukf_yaw)
    vel_wheel_y = vel_wheel_x_ori * np.sin(ukf_yaw) + vel_wheel_y_ori * np.cos(ukf_yaw)
    #if vel_wheel_x != 0 and vel_wheel_y!=0:
    #print 'after',vel_wheel_x,  vel_wheel_y ,'yaw',ukf_yaw

def callback_yaw(yaw):
    global ukf_yaw
    qn_ukf = (yaw.pose.pose.orientation.x, yaw.pose.pose.orientation.y, yaw.pose.pose.orientation.z, yaw.pose.pose.orientation.w)
    (ukf_roll,ukf_pitch,ukf_yaw) = euler_from_quaternion(qn_ukf)


rospy.init_node('ukf_predict_node')
UKFinit()
subimu = rospy.Subscriber('/map/imu/data', Imu, callback_imu)
subwheel = rospy.Subscriber('/robo/wheel/data', Vector3Stamped, callback_wheel)
subenemy = rospy.Subscriber('rgb_detection/enemy_position' Odometry, callback_enemy)

pub_ukf_vel = rospy.Publisher('/ukf/vel', Odometry, queue_size=1)
rospy.spin()