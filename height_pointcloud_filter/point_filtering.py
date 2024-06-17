#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
import tf
import tf.transformations
import tf2_ros
import threading


class PointCloudFilter:
    def __init__(self):
        # initialize the ROS node
        rospy.init_node('point_cloud_filter', anonymous=True)

        # Subscribe to the point cloud topic
        self.pc_subscriber = rospy.Subscriber("/registered_scan", PointCloud2, self.callback)
        
        # Subscribe to the Odometry topic
        self.odom_subscriber = rospy.Subscriber("/state_estimation", Odometry, self.odom_callback)

        # Publisher for the filtered point cloud
        self.publisher = rospy.Publisher("/filtered_points", PointCloud2, queue_size=10)

        # Set desired height threshold 
        self.height_threshhold = 0.5 

        # Initialize vehicle state variables 
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_z = 0.0
        self.vehicle_roll = 0.0
        self.vehicle_pitch = 0.0
        self.vehicle_yaw = 0.0

        self.sensor_offset_x = 0.0
        self.sensor_offset_y = 0.0
        self.vehicle_height = 0.75

        # Publish static transforms
        self.publish_static_transforms()

        # start dynamic transform broadcaster in a seperate thread
        self.dynamic_broadcaster_thread = threading.Thread(target=self.dynamic_transform_broadcaster)
        self.dynamic_broadcaster_thread.start()

    
    def publish_static_transforms(self):
        # Publish static transformations
        static_broadcaster = tf2_ros.StaticTransformBroadcaster()

        static_transformStamped = TransformStamped()
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "map"
        static_transformStamped.child_frame_id = 'velodyne'
        static_transformStamped.transform.translation.x = 0.0 
        static_transformStamped.transform.translation.y = 0.0  
        static_transformStamped.transform.translation.z = 0.0
        quat = tf.transformations.quaternion_from_euler(0, 0, 0)
        static_transformStamped.transform.rotation.x = quat[0]   
        static_transformStamped.transform.rotation.y = quat[1] 
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3] 

        static_broadcaster.sendTransform(static_transformStamped)


    def dynamic_transform_broadcaster(self):
        # Broadcast dynamic trasnforms
        dynamic_broadcaster = tf.TransformBroadcaster()
        rate = rospy.Rate(200)  # 200hz 

        while not rospy.is_shutdown():
            # Broadcast the vehicle's state
            self.update_vehicle_state()

            quat = tf.transformations.quaternion_from_euler(self.vehicle_roll, self.vehicle_pitch, self.vehicle_yaw)

            dynamic_broadcaster.sendTransform(
                (self.vehicle_x, self.vehicle_y, self.vehicle_z),
                quat,
                rospy.Time.now(),
                "velodyne",    # child frame
                "map"   # parent frame
            )

            # Broadcast terrain orientation
            # terrain_quat = tf.transformations.quaternion_from_euler(self.vehicle_roll, self.vehicle_pitch, 0)
            # dynamic_broadcaster.sendTransform(
                # (self.vehicle_x, self.vehicle_y, self.vehicle_z),
                # terrain_quat,
                # rospy.Time.now(),
                # "map",  # parent frame
                # "sensor"    # child frame
            # )

            rate.sleep()


    def odom_callback(self, msg):
        # Update vehicle state based on odometry data
        self.vehicle_x = msg.pose.pose.position.x
        self.vehicle_y = msg.pose.pose.position.y
        self.vehicle_z = msg.pose.pose.position.z

        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (self.vehicle_roll, self.vehicle_pitch, self.vehicle_yaw) = tf.transformations.euler_from_quaternion(orientation_list)
    

    def update_vehicle_state(self):
        pass


    def callback(self, msg):   
        # convert the message into a list of points
        # cloud = list(pc2.read_points(msg, skip_nans=True))

        # If the message is already a list, uncomment
        cloud = pc2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True)

        # Filter points based on the height threshold 
        filtered_points = [point for point in cloud if point.z <= self.height_threshhold]

        # Create a new PointCloud2 message with the filtered points
        filtered_cloud = pc2.create_cloud_xyz32(msg.header, filtered_points)

        # Publish the filtered point cloud
        self.publisher.publish(filtered_cloud)

        # Debugging
        rospy.loginfo("published filtered cloud")



if __name__ == "__main__":
    # create an instance of the class
    filter_node = PointCloudFilter()

    # Keep the node running
    rospy.spin()