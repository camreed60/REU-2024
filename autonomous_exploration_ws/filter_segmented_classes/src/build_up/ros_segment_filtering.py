#!/usr/bin/env python3

import rospy
import numpy as np
from ultralytics import YOLO
import cv2
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import TwistStamped
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg 
import tf
import tf2_ros
import threading
import math

# Camera parameters
fx = 205.47  # Focal length in x direction (example value) # 205.47 in sim # rs 322.28
fy = 205.47 # Focal length in y direction (example value) # 205.47 in sim # rs 322.28
cx = 320.5 # Principal point x (example value) # 320.5 in sim # rs 320.82
cy = 180.5  # Principal point y (example value) # 180.5 in sim # rs 178.77

# model weights
model_path = "/home/wvuirl/ros_environment/autonomous_exploration_ws/src/filter_segmented_classes/src/walkway_v1.pt"

class SegmentationPointCloud:
    def __init__(self):
        # start the bridge
        self.bridge = CvBridge()

        # load the model
        self.model = YOLO(model_path)

        # initialize publisher topic
        self.pointcloud_pub = rospy.Publisher('/segmented_pointcloud', PointCloud2, queue_size=10)

        # initialize subscriber topics
        rospy.Subscriber('/rgbd_camera/depth/image', Image, self.depth_callback) # change
        rospy.Subscriber('/rgbd_camera/color/image', Image, self.rgb_callback) # change
        rospy.Subscriber('/state_estimation', Odometry, self.odom_callback)
        rospy.Subscriber('/cmd_vel', TwistStamped, self.speed_callback) # change?

        # initialize variables
        self.rgb_image = None
        self.depth_image = None

        # Initialize vehicle state variables
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_z = 0.0
        self.vehicle_roll = 0.0
        self.vehicle_pitch = 0.0
        self.vehicle_yaw = 0.0

        self.vehicle_speed = 0.0
        self.vehicle_yaw_rate = 0.0

        self.sensor_offset_x = 0.0
        self.sensor_offset_y = 0.0
        self.vehicle_height = 0.75

        # Publish static transforms
        self.publish_static_transforms()

        # Start dynamic transform broadcaster in a separate thread
        self.dynamic_broadcaster_thread = threading.Thread(target=self.dynamic_transform_broadcaster)
        self.dynamic_broadcaster_thread.start()


    def depth_callback(self, msg):
        # get the depth image
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # threading.Thread(target=self.create_colored_pointcloud).start()  # Added threading
        # remove threading?
    def rgb_callback(self, msg):
        # get the rgb image
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # threading.Thread(target=self.create_colored_pointcloud).start()  # Added threading

    def odom_callback(self, msg):
        # update the vehicle state based on odometry data
        self.vehicle_x = msg.pose.pose.position.x
        self.vehicle_y = msg.pose.pose.position.y
        self.vehicle_z = msg.pose.pose.position.z

        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (self.vehicle_roll, self.vehicle_pitch, self.vehicle_yaw) = tf.transformations.euler_from_quaternion(orientation_list)
    

    def speed_callback(self, msg):
        self.vehicle_speed = msg.twist.linear.x
        self.vehicle_yaw_rate = msg.twist.angular.z


    def publish_static_transforms(self):
        # Publish static transformations
        static_broadcaster = tf2_ros.StaticTransformBroadcaster()

        static_transformStamped = TransformStamped()
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "map"
        static_transformStamped.child_frame_id = 'corrected_camera' # update this with the correct child frame
        static_transformStamped.transform.translation.x = 0.0 
        static_transformStamped.transform.translation.y = 0.0  
        static_transformStamped.transform.translation.z = 0.0

        # change these to change projection orientation
        roll = 0.0
        pitch = 0.0# 90 degree rotation around y-axis
        yaw = 0.0

        quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        static_transformStamped.transform.rotation.x = quat[0]   
        static_transformStamped.transform.rotation.y = quat[1] 
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3] 

        static_broadcaster.sendTransform(static_transformStamped)


    def dynamic_transform_broadcaster(self):
        # Broadcast dynamic transforms
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
                "corrected_camera",   # corrected 
                "map"   
            )

            rate.sleep()

    def update_vehicle_state(self):
        # Update the vehicle's position and orientation based on speed and yaw rate
        dt = 0.005  # Assume a time step of 5 milliseconds

        # Update yaw
        self.vehicle_yaw += dt * self.vehicle_yaw_rate
        if self.vehicle_yaw > math.pi:
            self.vehicle_yaw -= 2 * math.pi
        elif self.vehicle_yaw < -math.pi:
            self.vehicle_yaw += 2 * math.pi

        # Update position
        self.vehicle_x += dt * math.cos(self.vehicle_yaw) * self.vehicle_speed + dt * self.vehicle_yaw_rate * (-math.sin(self.vehicle_yaw) * self.sensor_offset_x - math.cos(self.vehicle_yaw) * self.sensor_offset_y)
        self.vehicle_y += dt * math.sin(self.vehicle_yaw) * self.vehicle_speed + dt * self.vehicle_yaw_rate * (math.cos(self.vehicle_yaw) * self.sensor_offset_x - math.sin(self.vehicle_yaw) * self.sensor_offset_y)
        self.vehicle_z = self.vehicle_height


    def get_segmentation_masks_and_classes(self, img):
        # Perform inference
        results = self.model.predict(img)

        # Extract class names from the model
        class_names = self.model.names

        # Extract segmentation masks and class labels
        binary_masks = []
        class_labels = []
        if results[0].masks is not None:
            masks = results[0].masks.data
            for mask, cls in zip(masks, results[0].boxes.cls):
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                mask_np = cv2.resize(mask_np, (self.depth_image.shape[1], self.depth_image.shape[0]))  # Resize mask to match depth image
                binary_masks.append(mask_np.flatten()) # .reshape(-1)
                class_labels.append(int(cls.item()))

        return binary_masks, class_labels, class_names


    def rgbd_to_pointcloud(self, depth_image):
        height, width = depth_image.shape
        pointcloud = np.zeros((height, width, 3))

        for v in range(height):
            for u in range(width):
                Z = depth_image[v, u]
                if Z > 0:  # Valid depth value
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    pointcloud[v, u] = [X, Y, Z]

        return pointcloud.reshape(-1, 3)


    def create_colored_pointcloud(self):
        if self.rgb_image is None or self.depth_image is None:
            return
        
        pointcloud = self.rgbd_to_pointcloud(self.depth_image)
        binary_masks, class_labels, class_names = self.get_segmentation_masks_and_classes(self.rgb_image)

        # initialize color map based on classes
        color_map = {
        0: [1.0, 0.0, 0.0],  # Red for class 0 : grass
        1: [1.0, 1.0, 0.0],  # Yellow for class 1 : gravel
        2: [0.5, 0.0, 0.5],  # Purple for class 2 : mulch
        3: [1.0, 0.5, 0.0],  # Orange for class 3 : obstacle
        4: [0.0, 0.0, 1.0],  # Blue for class 4 : parking lot
        5: [0.0, 1.0, 0.0],  # Green for class 5 : sidewalk
        6: [0.5, 0.5, 0.5],  # Gray for class 6 : unused class
        7: [1.0, 0.0, 1.0]   # Magenta for class 7 : vegetation
        }

        # Initialize the color array with black
        colors = np.zeros_like(pointcloud)  

        # Apply colors based on masks and their classes
        for mask, cls in zip(binary_masks, class_labels):
            color = color_map.get(cls, [1.0, 1.0, 1.0])  # Default to white if class not in color_map
            colors[mask == 255] = color

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'rgbd_camera'
        pointcloud_msg = pc2.create_cloud_xyz32(header, pointcloud)
        pointcloud_msg.fields.append(pc2.PointField(name='rgb', offset=12, datatype=7, count=1))
        pointcloud_msg.is_dense = False

        self.pointcloud_pub.publish(pointcloud_msg)


if __name__ == '__main__':
    rospy.init_node('segmentation_pointcloud_node', anonymous=True)
    spc = SegmentationPointCloud()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        spc.create_colored_pointcloud()
        rate.sleep()
