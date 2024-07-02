#!/usr/bin/env python3

import rospy
import numpy as np
from ultralytics import YOLO
import cv2
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg 
import open3d as o3d
from scipy.interpolate import griddata
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation

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
        
        self.pointcloud_pub = rospy.Publisher('/segmented_pointcloud', PointCloud2, queue_size=10)

        # initialize subscriber topics
        depth_sub = Subscriber('/rgbd_camera/depth/image', Image) # change
        rgb_sub = Subscriber('/rgbd_camera/color/image', Image) # change

        # Synchronize rgb and depth callback
        ts = ApproximateTimeSynchronizer([depth_sub, rgb_sub], 10, 0.1)
        ts.registerCallback(self.synchronized_callback)

        # start listener
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.source_frame = 'rgbd_camera'
        self.target_frame = 'map'
        

        
    def synchronized_callback(self, depth_msg, rgb_msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        self.create_colored_pointcloud()

    def get_segmentation_masks_and_classes(self, img):
        # Perform inference
        results = self.model.predict(img)

        # Extract class names from the model
        class_names = self.model.names


        # Extract segmentation masks and class labels
        binary_masks, class_labels = [], []
        if results[0].masks is not None:
            masks = results[0].masks.data
            classes = results[0].boxes.cls
            masks_np = masks.cpu().numpy().astype(np.uint8)
            masks_np = np.array([cv2.resize(mask, (self.depth_image.shape[1], self.depth_image.shape[0])) for mask in masks_np])
            binary_masks = masks_np.reshape(masks_np.shape[0], -1)
            class_labels = classes.cpu().numpy().astype(int)
        return binary_masks, class_labels, class_names


    def rgbd_to_pointcloud(self, depth_image, fx, fy, cx, cy):
        

        # create a mesh grid
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        Z = depth_image
        X = (u-cx) * Z / (fx)
        Y = (v-cy) * Z / (fy)

        return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)


    def create_colored_pointcloud(self):
        # downsample depth image
        downsampled_depth = cv2.resize(self.depth_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

        # adjust camera [arameters for downsampled image
        downsampled_fx, downsampled_fy = fx * 0.5, fy * 0.5
        downsampled_cx, donwsampled_cy = cx * 0.5, cy * 0.5

        # create a pointcloud from the depth image
        pointcloud = self.rgbd_to_pointcloud(downsampled_depth, downsampled_fx, downsampled_fy, downsampled_cx, donwsampled_cy)

        # retrieve masks and their respective class
        binary_masks, class_labels, _ = self.get_segmentation_masks_and_classes(self.rgb_image)

        # set target class to be filtered
        full_size_mask = np.any(binary_masks[class_labels == 0], axis=0)
        full_size_mask = full_size_mask.reshape(self.rgb_image.shape[:2])

        # downsample the mask to match depth image size
        target_class_mask = cv2.resize(full_size_mask.astype(np.uint8), (downsampled_depth.shape[1], downsampled_depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # apply mask to pointcloud
        pointcloud = pointcloud[target_class_mask.flatten()]

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        pointcloud_msg = pc2.create_cloud_xyz32(header, pointcloud)
        
        # transform to current frame and publish the new pointcloud 
        transformed_pointcloud_msg = self.transform_pointcloud(pointcloud_msg, self.source_frame, self.target_frame)
        self.pointcloud_pub.publish(transformed_pointcloud_msg)

    def transform_pointcloud(self, pointcloud, source_frame, target_frame):
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
            
            # Convert the transform to a 4x4 matrix
            transform_matrix = self.transform_to_matrix(trans.transform)
            
            # Extract points from the pointcloud message
            points = np.array(list(pc2.read_points(pointcloud, field_names=("x", "y", "z"), skip_nans=True)))
            
            # Check if points is empty
            if points.size == 0:
                rospy.logwarn("Received an empty point cloud")
                return pointcloud  # Return the original pointcloud if it's empty
            
            # Ensure points is 2D
            if points.ndim == 1:
                points = points.reshape(-1, 3)
            
            # Add homogeneous coordinate
            points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
            
            # Apply transformation
            transformed_points = np.dot(points_homogeneous, transform_matrix.T)[:, :3]

            # Create new pointcloud message
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = target_frame
            transformed_pointcloud = pc2.create_cloud_xyz32(header, transformed_points)
            
            return transformed_pointcloud
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Error transforming point cloud: {e}")
            return pointcloud

    def transform_to_matrix(self, transform):
        translation = transform.translation
        rotation = transform.rotation
        
        # Convert quaternion to rotation matrix using scipy
        r = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
        matrix = np.eye(4)
        matrix[:3, :3] = r.as_matrix()
        
        # Add translation
        matrix[0:3, 3] = [translation.x, translation.y, translation.z]
        
        return matrix


if __name__ == '__main__':
    rospy.init_node('segmentation_pointcloud_node', anonymous=True)
    spc = SegmentationPointCloud()
    rospy.spin()
