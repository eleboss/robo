#!/usr/bin/python2.7
# coding=<encoding name> 例如，可添加# coding=utf-8

###############UKF参数选取和系统方程测定方法##############
#参数选取原则  
#首先按照动力学方程建立F方程  这里我使用的是匀速运动方程，考虑到引入加速度可能会造成不稳定，而且匀速方程表现较好，因此使用匀速方程。 S（i+1） = S（i） + V*dt +0.5 a * dt*dt
#F要和H相乘，H=【px vx ax py vy ay】，所以写为下面的形式，两个yaw分别是imu和uwb的观测
#用H来选择有几个输入参数，这里我有三个参数，分别是yaw观测1，yaw观测2，dyaw。  
#dim_x = 3， 表示输入方程有三个输入  
#dim_z = 3 表示观测方程z有3个观测。yaw，dyaw 
#p_std_x, v_std_x，a_std_x，表示yz两轴的测量误差，这里根据测量结果填写，p、v、a、对应 位置，速度，加速度
#Q噪声项，var这里我发现用0.2的结果比较好，目前还不知道为什么  
#P里头分别填上前面观测能达到的最大值。【px vx ax py vy ay】
#ukf.x = np.array([0., 0., 0., 0. ,0., 0.])表示起始位置和起始速度都是0
# MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-1.0) 6表示[px vx ax py vy ay】里头有6个
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
# 


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



imu_init = True
vel_wheel_x = vel_wheel_y = 0
pos_uwb_x = pos_uwb_y = uwb_seq = last_uwb_seq = uwb_time = 0
ukf_vel_x = ukf_vel_y = 0
ukf_yaw = 0
vel_wheel_x = vel_wheel_y = 0

ukf_result = []
qn_ukf = [0,0,0,0]

# S（i+1） = S（i） + V*dt + 0.5*a*dt*dt
def f_cv(x, dt):
    """ state transition function for a 
    constant velocity aircraft"""
    
    F = np.array([[1, dt, 0.5*dt*dt, 0,  0,          0],
                  [0,  1, dt,        0,  0,          0],
                  [0,  0, 1,         0,  0,          0],
                  [0,  0, 0,         1, dt,  0.5*dt*dt],
                  [0,  0, 0,         0,  1,         dt],
                  [0,  0, 0,         0,  0,          1]], dtype=float)
    return np.dot(F, x)

def h_cv(x):
    return np.array([x[0], x[1], x[2], x[3], x[4], x[5]])

# 信号源
# p_std_x，p_std_y：融合位置信息
# v_std_x, v_std_y：UKF融合轮式里程计和imu后输出速度
# a_std_x, a_std_y：imu输出加速度

def UKFinit():
    global ukf
    ukf_fuse = []
    p_std_x, p_std_y = 0.1, 0.1
    v_std_x, v_std_y = 0.01, 0.01
    a_std_x, a_std_y = 0.01, 0.01
    dt = 0.0125 #80HZ


    sigmas = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-1.0)
    ukf = UKF(dim_x=6, dim_z=6, fx=f_cv, hx=h_cv, dt=dt, points=sigmas)
    ukf.x = np.array([0., 0., 0., 0., 0., 0.])
    ukf.R = np.diag([p_std_x, v_std_x, a_std_x, p_std_y, v_std_y, a_std_y]) 
    ukf.Q[0:3, 0:3] = Q_discrete_white_noise(3, dt=dt, var=0.2)
    ukf.Q[3:6, 3:6] = Q_discrete_white_noise(3, dt=dt, var=0.2)
    ukf.P = np.diag([8, 2, 2, 5, 2, 2])



def callback_imu(imu):
    global imu_last_time, imu_init, acc_imu_x, acc_imu_y, vel_wheel_x, vel_wheel_y, qn_ukf,ukf_yaw
    global ukf_result, ukf, pos_uwb_x, pos_uwb_y, uwb_seq, last_uwb_seq, uwb_time, incremental_pos_x, incremental_pos_y
    if imu_init == True:
        print "ukf process Init Finished!"
        imu_last_time = imu.header.stamp.secs + imu.header.stamp.nsecs * 10**-9
        vel_imu_x = 0
        vel_imu_y = 0
        incremental_pos_x = 0
        incremental_pos_y = 0
        imu_init = False
    else:
        imu_time = imu.header.stamp.secs + imu.header.stamp.nsecs * 10**-9

        acc_imu_x = imu.linear_acceleration.x
        acc_imu_y = imu.linear_acceleration.y

        dt = imu_time - imu_last_time

        # uwb not change, use imu to compensate
        if uwb_seq - last_uwb_seq == 0:
            incremental_pos_x = incremental_pos_x + vel_wheel_x * dt + 0.5 * acc_imu_x * dt * dt
            pos_fuse_x = pos_uwb_x + incremental_pos_x

            incremental_pos_y = incremental_pos_y + vel_wheel_y * dt + 0.5 * acc_imu_y * dt * dt
            pos_fuse_y = pos_uwb_y + incremental_pos_y
        else:
            dt = imu_time - uwb_time
            incremental_pos_x = 0
            incremental_pos_x = incremental_pos_x + vel_wheel_x * dt + 0.5 * acc_imu_x * dt * dt
            pos_fuse_x = pos_uwb_x + incremental_pos_x

            incremental_pos_y = 0
            incremental_pos_y = incremental_pos_y + vel_wheel_y * dt + 0.5 * acc_imu_y * dt * dt
            pos_fuse_y = pos_uwb_y + incremental_pos_y
        
        #print 'uwb',pos_uwb_x, pos_uwb_y, 'wheel',vel_wheel_x, vel_wheel_y, 'imu', acc_imu_x, acc_imu_y, 'yaw',ukf_yaw
        #if acc_imu_x > 1 or acc_imu_y > 1:
        #    print 'imu', acc_imu_x, acc_imu_y
        imu_last_time = imu_time
        last_uwb_seq = uwb_seq

        ukf_input = [pos_fuse_x, vel_wheel_x, acc_imu_x, pos_fuse_y, vel_wheel_y, acc_imu_y]
        ukf.predict()
        ukf.update(ukf_input)
        ukf_out_pos_x = ukf.x[0]
        ukf_out_pos_y = ukf.x[3]
        #print ukf.x
        print 'UWB x:',pos_uwb_x,'UWB y:',pos_uwb_y
        print 'FUSE x',pos_fuse_x,'FUSE y',pos_fuse_y
        print 'KALMAN x',ukf_out_pos_x,'KALMAN y',ukf_out_pos_y

        ukf_pos = Odometry()
        ukf_pos.header.frame_id = "ukf_pos"
        ukf_pos.header.stamp.secs = imu.header.stamp.secs
        ukf_pos.header.stamp.nsecs = imu.header.stamp.nsecs

        ukf_pos.pose.pose.orientation.x = qn_ukf[0]
        ukf_pos.pose.pose.orientation.y = qn_ukf[1]
        ukf_pos.pose.pose.orientation.z = qn_ukf[2]
        ukf_pos.pose.pose.orientation.w = qn_ukf[3]

        ukf_pos.pose.pose.position.x = pos_fuse_x
        ukf_pos.pose.pose.position.y = pos_fuse_y
        
        ukf_pos.twist.twist.linear.x = ukf_vel_x
        ukf_pos.twist.twist.linear.y = ukf_vel_y

        pub_ukf_pos.publish(ukf_pos)






def callback_uwb(uwb):
    global pos_uwb_x, pos_uwb_y , uwb_seq, uwb_time

    uwb_seq = uwb.header.seq
    uwb_time = uwb.header.stamp.secs + uwb.header.stamp.nsecs * 10**-9

    pos_uwb_x = uwb.pose.pose.position.x
    pos_uwb_y = uwb.pose.pose.position.y

#def callback_vel(vel):
#    global ukf_vel_x, ukf_vel_y
#    ukf_vel_x = vel.twist.twist.linear.x
#    ukf_vel_y = vel.twist.twist.linear.y 

def callback_yaw(yaw):
    global qn_ukf, ukf_yaw
    #only yaw are available
    qn_ukf = [yaw.pose.pose.orientation.x, yaw.pose.pose.orientation.y, yaw.pose.pose.orientation.z, yaw.pose.pose.orientation.w]
    (ukf_roll,ukf_pitch,ukf_yaw) = euler_from_quaternion(qn_ukf)

    
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


rospy.init_node('ukf_process_node')
UKFinit()
subimu = rospy.Subscriber('/map/imu/data', Imu, callback_imu)
subuwb = rospy.Subscriber('/map/uwb/data', Odometry, callback_uwb)
#subvel = rospy.Subscriber('/ukf/vel', Odometry, callback_vel)
subwheel = rospy.Subscriber('/robo/wheel/data', Vector3Stamped, callback_wheel)
subyaw = rospy.Subscriber('/ukf/yaw', Odometry, callback_yaw)

pub_ukf_pos = rospy.Publisher('/ukf/pos', Odometry, queue_size=1)
rospy.spin()