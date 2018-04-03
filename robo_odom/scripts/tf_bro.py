#!/usr/bin/python2.7
import roslib  
#roslib.load_manifest('learning_tf')  
import rospy   
import tf
import tf2_ros

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion

#use to rotate imu axis to uwb axis
ROTATION_BAIS = 0
ukf_x = ukf_y = ukf_z = 0
ukf_yaw = 0
imu_roll = imu_pitch = imu_yaw = 0
def handle_pose(msg, dronename, yaw, roll, pitch): 
    global ukf_yaw, imu_roll, imu_pitch


def callback_ukf(yaw):
    global ukf_yaw, imu_roll, imu_pitch, imu_yaw, ROTATION_BAIS
    qn_ukf =  yaw.pose.pose.orientation.x, yaw.pose.pose.orientation.y, yaw.pose.pose.orientation.z, yaw.pose.pose.orientation.w
    (ukf_roll,ukf_pitch,ukf_yaw) = euler_from_quaternion(qn_ukf)

	#ues imu_yaw - ukf_yaw for axis fix. and ROTATION_BAIS to rotate the imu axis to uwb global axis
    (x, y, z, w) = quaternion_from_euler(0, 0, ukf_yaw - imu_yaw   + ROTATION_BAIS)
    

    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()
    
    # Here we use the ukf yaw as the yaw, ukf yaw is the yaw in map axis.
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = 'uwb_link'
    t.child_frame_id = 'robo_link'
    t.transform.translation.x = 0
    t.transform.translation.y = 0
    t.transform.translation.z = 0
    t.transform.rotation.x = x
    t.transform.rotation.y = y
    t.transform.rotation.z = z
    t.transform.rotation.w = w
    
    br.sendTransform(t) 



def callback_imu(imu):
    global imu_roll, imu_pitch, imu_yaw
    qn_imu = [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]
    (imu_roll,imu_pitch,imu_yaw) = euler_from_quaternion(qn_imu)
    



if __name__ == '__main__':  
    rospy.init_node('turtle_tf_broadcaster')  

    usb_ukf_yaw = rospy.Subscriber('/ukf/yaw', Odometry, callback_ukf)
    subimu = rospy.Subscriber('/robo/imu/data', Imu, callback_imu)

    rospy.spin()  
