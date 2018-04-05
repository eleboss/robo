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
import tf
import tf2_ros
from numpy.random import randn
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from robo_perception.msg import ObjectList
from robo_perception.msg import Object
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped

PREDICT_INIT = True

ukf_result = []
robo_vel_x = robo_vel_y = 0
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

    p_std_x, p_std_y = 0.1, 0.1
    v_std_x, v_std_y = 0.1, 0.1
    dt = 0.0125 #80HZ


    sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1.0)
    ukf = UKF(dim_x=4, dim_z=4, fx=f_cv, hx=h_cv, dt=dt, points=sigmas)
    ukf.x = np.array([0., 0., 0., 0.])
    ukf.R = np.diag([p_std_x, v_std_x, p_std_y, v_std_y]) 
    ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.2)
    ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.2)
    ukf.P = np.diag([8, 1.5 ,5, 1.5])


def callback_enemy(enemy):
    global last_enemy_num, PREDICT_INIT, robo_vel_x, robo_vel_y
    global ukf_result, ukf, max_move_distance, aim_target_x, aim_target_y, found_aim, aim_lost
    if PREDICT_INIT == True:
        print "ukf vel Init Finished!"
        enemy_last_time = enemy.header.stamp.secs + enemy.header.stamp.nsecs * 10**-9
        last_enemy_num = 0
        max_move_distance = 0.5
        lost_time = []
        PREDICT_INIT = False
        found_aim = False
        aim_lost = False
    else:
        enemy_time = imu.header.stamp.secs + imu.header.stamp.nsecs * 10**-9
        for i in range(enemy.num):
            object_name =  enemy.object[i].team.data
            if object_name == 'red0' or object_name == 'red1':
                enemy_num = enemy_num + 1
                try:
                    enemy_object_trans.append(tfBuffer.lookup_transform(object_name, 'base_link', rospy.Time(0), rospy.Duration(0.1)))
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    print('ENEMY OBJ TRANS FAIL! check your code')
            elif object_name == 'blue0':
                team_num = team_num + 1
                try:
                    team_object_trans.append(tfBuffer.lookup_transform(object_name, 'base_link', rospy.Time(0), rospy.Duration(0.1)))
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    print('TEAM OBJ TRANS FAIL! check your code')

        #object judgement
        if enemy_num == 1
            enemy_0_x = enemy_object_trans[0].transform.translation.x
            enemy_0_y = enemy_object_trans[0].transform.translation.y
            enemy_1_x = 99
            enemy_1_y = 99
        elif enemy_num == 2
            enemy_0_x = enemy_object_trans[0].transform.translation.x
            enemy_0_y = enemy_object_trans[0].transform.translation.y
            enemy_1_x = enemy_object_trans[1].transform.translation.x
            enemy_1_y = enemy_object_trans[1].transform.translation.y

        dt = enemy_time - enemy_last_time   
        if enemy_num == 0 and last_enemy_num == 0:
            found_aim = False
            if aim_lost == True or lost_counter < 15:
                print 'Lost OBJ'
                lost_counter = lost_counter + 1 
                lost_time.append[dt_0]
            else:
                print 'NO ENEMY!'
                lost_counter = 0
        elif enemy_num == 1 and last_enemy_num == 0:
            print 'Found enemy!!'
            euclid_distance_0 = np.sqrt((enemy_0_x - aim_target_x) ** 2 + (enemy_0_y - aim_target_y) ** 2)
            if aim_lost == True:
                if euclid_distance_0 > 0.5:
                    print 'Find an enemy, but not the targted one.'
                    found_aim = False
                    aim_lost = True
                else:
                    found_aim = True
                    aim_lost = False
                    dt = np.sum(lost_time)
                    enemy_vel_x = (enemy_0_x - last_enemy_0_x) / dt
                    enemy_vel_y = (enemy_0_y - last_enemy_0_y) / dt
                    ukf_input = [enemy_0_x, enemy_0_vel_x, enemy_0_y, enemy_0_vel_y]
                    ukf.predict()
                    ukf.update(ukf_input)
            else:
                if euclid_distance_0 > 0.5:
                    print 'Find an enemy, but not the targted one.'
                    found_aim = False
                    aim_lost = False
                else:
                    found_aim = True
                    aim_lost = False
                    ukf_input = [enemy_0_x, 0, enemy_0_y, 0]
                    ukf.predict()
                    ukf.update(ukf_input)
                print 'not detect the targeted one'
        elif enemy_num ==2 and last_enemy_num == 0:
            print 'Found enemy!!'
            euclid_distance_0 = np.sqrt((enemy_0_x - aim_target_x) ** 2 + (enemy_0_y - aim_target_y) ** 2)
            euclid_distance_1 = np.sqrt((enemy_1_x - aim_target_x) ** 2 + (enemy_1_y - aim_target_y) ** 2)
            if euclid_distance_0 > 0.5 and euclid_distance_1 > 0.5:
                print 'Diveration too large, somthing wrong, check all, unable to predict!!!!'
                found_aim = False
            else:
                found_aim == True
                if euclid_distance_1 < euclid_distance_0:
                    dt_0 = np.sum(lost_time_0)
                    enemy_vel_x = (enemy_0_x - last_enemy_0_x) / dt
                    enemy_vel_y = (enemy_0_y - last_enemy_0_y) / dt
                    ukf_input = [enemy_0_x, enemy_0_vel_x, enemy_0_y, enemy_0_vel_y]
                    ukf.predict()
                    ukf.update(ukf_input)
                else:
                    enemy_vel_x = (last_enemy_1_x - enemy_1_x) / dt
                    enemy_vel_y = (last_enemy_1_y - enemy_1_y) / dt
                    ukf_input = [enemy_1_x, enemy_vel_x, enemy_1_y, enemy_vel_y]
                    ukf.predict()
                    ukf.update(ukf_input)
        elif enemy_num == 0 and last_enemy_num != 0:
            print 'Found enemy!!'
            if found_aim == True:
                found_aim = False
                aim_lost = True
                lost_counter = lost_counter + 1 
                lost_time.append[dt]
        elif enemy_num == 1 and last_enemy_num == 2:
            print 'Found enemy!!'
            euclid_distance_0 = np.sqrt((enemy_0_x - aim_target_x) ** 2 + (enemy_0_y - aim_target_y) ** 2)
            if euclid_distance_0 > 0.5:
                print 'Diveration too large, aming object may not appear in realsense camera, lost object'
                found_aim = False
                aim_lost = True
                lost_counter = lost_counter + 1 
                lost_time.append[dt]
                ukf.predict()                
            else:
                found_aim == True
                dt_0 = np.sum(lost_time_0)
                enemy_vel_x = (enemy_0_x - last_enemy_0_x) / dt
                enemy_vel_y = (enemy_0_y - last_enemy_0_y) / dt
                ukf_input = [enemy_0_x, enemy_0_vel_x, enemy_0_y, enemy_0_vel_y]
                ukf.predict()
                ukf.update(ukf_input)
        elif enemy_num == 2 and last_enemy_num == 2:
            print 'Found enemy!!'
            euclid_distance_0 = np.sqrt((enemy_0_x - aim_target_x) ** 2 + (enemy_0_y - aim_target_y) ** 2)
            euclid_distance_1 = np.sqrt((enemy_1_x - aim_target_x) ** 2 + (enemy_1_y - aim_target_y) ** 2)
            found_aim = True
            if euclid_distance_1 < euclid_distance_0:
                dt_0 = np.sum(lost_time_0)
                enemy_vel_x = (enemy_0_x - last_enemy_0_x) / dt
                enemy_vel_y = (enemy_0_y - last_enemy_0_y) / dt
                ukf_input = [enemy_0_x, enemy_0_vel_x, enemy_0_y, enemy_0_vel_y]
                ukf.predict()
                ukf.update(ukf_input)
            else:
                enemy_vel_x = (last_enemy_1_x - enemy_1_x) / dt
                enemy_vel_y = (last_enemy_1_y - enemy_1_y) / dt
                ukf_input = [enemy_1_x, enemy_vel_x, enemy_1_y, enemy_vel_y]
                ukf.predict()
                ukf.update(ukf_input)
        elif enemy_num == 1 and last_enemy_num == 1:
            euclid_distance_0 = np.sqrt((enemy_0_x - aim_target_x) ** 2 + (enemy_0_y - aim_target_y) ** 2)
            if found_aim == True:
                if euclid_distance_0 > 0.5:
                    found_aim = False
                    aim_lost = True
                    lost_counter = lost_counter + 1 
                    lost_time.append[dt]
                    ukf.predict()   
                else:
                    found_aim = True
                    enemy_vel_x = (enemy_0_x - last_enemy_0_x) / dt
                    enemy_vel_y = (enemy_0_y - last_enemy_0_y) / dt
                    ukf_input = [enemy_0_x, enemy_0_vel_x, enemy_0_y, enemy_0_vel_y]
                    ukf.predict()
                    ukf.update(ukf_input)
            else:
                if euclid_distance_0 > 0.5:
                    found_aim = False
                    aim_lost = True
                    lost_counter = lost_counter + 1 
                    lost_time.append[dt]
                    ukf.predict()   
                else:
                    found_aim = True
                    aim_lost = False
                    dt = np.sum(lost_time)
                    enemy_vel_x = (enemy_0_x - last_enemy_0_x) / dt
                    enemy_vel_y = (enemy_0_y - last_enemy_0_y) / dt
                    ukf_input = [enemy_0_x, enemy_0_vel_x, enemy_0_y, enemy_0_vel_y]
                    ukf.predict()
                    ukf.update(ukf_input)

        elif enemy_num == 2 and last_enemy_num == 1:
            euclid_distance_0 = np.sqrt((enemy_0_x - aim_target_x) ** 2 + (enemy_0_y - aim_target_y) ** 2)
            euclid_distance_1 = np.sqrt((enemy_1_x - aim_target_x) ** 2 + (enemy_1_y - aim_target_y) ** 2)
            if found_aim = True:
                found_aim = True
                aim_lost = False 
                if euclid_distance_1 < euclid_distance_0:
                    enemy_vel_x = (enemy_0_x - last_enemy_0_x) / dt
                    enemy_vel_y = (enemy_0_y - last_enemy_0_y) / dt
                    ukf_input = [enemy_0_x, enemy_0_vel_x, enemy_0_y, enemy_0_vel_y]
                    ukf.predict()
                    ukf.update(ukf_input)
                else:
                    enemy_vel_x = (last_enemy_1_x - enemy_1_x) / dt
                    enemy_vel_y = (last_enemy_1_y - enemy_1_y) / dt
                    ukf_input = [enemy_1_x, enemy_vel_x, enemy_1_y, enemy_vel_y]
                    ukf.predict()
                    ukf.update(ukf_input)
            else:
                euclid_distance_0 = np.sqrt((enemy_0_x - aim_target_x) ** 2 + (enemy_0_y - aim_target_y) ** 2)
                euclid_distance_1 = np.sqrt((enemy_1_x - aim_target_x) ** 2 + (enemy_1_y - aim_target_y) ** 2)
                found_aim = True
                aim_lost = False
                if euclid_distance_1 < euclid_distance_0:
                    dt = np.sum(lost_time)
                    enemy_vel_x = (enemy_0_x - last_enemy_0_x) / dt
                    enemy_vel_y = (enemy_0_y - last_enemy_0_y) / dt
                    ukf_input = [enemy_0_x, enemy_0_vel_x, enemy_0_y, enemy_0_vel_y]
                    ukf.predict()
                    ukf.update(ukf_input)
                else:
                    dt = np.sum(lost_time)
                    enemy_vel_x = (last_enemy_1_x - enemy_1_x) / dt
                    enemy_vel_y = (last_enemy_1_y - enemy_1_y) / dt
                    ukf_input = [enemy_1_x, enemy_vel_x, enemy_1_y, enemy_vel_y]
                    ukf.predict()
                    ukf.update(ukf_input)
            lost_counter = 0
            lost_time = []
        last_enemy_num = enemy_num
        enemy_last_time = enemy_time


        ukf_out_pos_x = ukf.x[0]
        ukf_out_vel_x = ukf.x[1]
        ukf_out_pos_x = ukf.x[2]
        ukf_out_vel_y = ukf.x[3]

        ukf_vel = Odometry()
        ukf_vel.header.frame_id = "ukf_vel"
        ukf_vel.header.stamp.secs = imu.header.stamp.secs
        ukf_vel.header.stamp.nsecs = imu.header.stamp.nsecs
        ukf_vel.twist.twist.linear.x = ukf_out_vel_x
        ukf_vel.twist.twist.linear.y = ukf_out_vel_y
        pub_ukf_vel.publish(ukf_vel)


def callback_ukf(ukf):
    global robo_vel_x, robo_vel_y
    robo_vel_x =  ukf..twist.twist.linear.x
    robo_vel_y =  ukf..twist.twist.linear.y




rospy.init_node('ukf_predict_node')
UKFinit()
subenemy = rospy.Subscriber('rgb_detection/enemy_position' ObjectList, callback_enemy)
subukf = rospy.Subscriber('/ukf/pos', Odometry, callback_ukf)

pub_ukf_vel = rospy.Publisher('/ukf/enemy', ObjectList, queue_size=1)
rospy.spin()