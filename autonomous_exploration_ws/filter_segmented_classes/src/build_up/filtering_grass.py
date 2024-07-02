#!/usr/bin/env python3

import rospy
import numpy as np
from ultralytics import YOLO
import cv2
import torch
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import TwistStamped
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg 
import tf
import open3d as o3d
import math
import time
from scipy.interpolate import griddata

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
        # self.model.to('cuda')
        # initialize publisher topic
        self.pointcloud_pub = rospy.Publisher('/segmented_pointcloud', PointCloud2, queue_size=10)

        # initialize subscriber topics
        rospy.Subscriber('/rgbd_camera/depth/image', Image, self.depth_callback) # change
        rospy.Subscriber('/rgbd_camera/color/image', Image, self.rgb_callback) # change

        # initialize variables
        self.rgb_image = None
        self.depth_image = None

        # start listener
        self.tf_listener = tf.TransformListener()

        self.source_frame = 'rgbd_camera'
        self.target_frame = 'map'
        

        
    def depth_callback(self, msg):
        # get the depth image
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # threading.Thread(target=self.create_colored_pointcloud).start()  # Added threading

    def rgb_callback(self, msg):
        # get the rgb image
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # threading.Thread(target=self.create_colored_pointcloud).start()  # Added threading

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
                mask_np = mask.cpu().numpy().astype(np.uint8)
                mask_np = cv2.resize(mask_np, (self.depth_image.shape[1], self.depth_image.shape[0]))  # Resize mask to match depth image
                binary_masks.append(mask_np.flatten()) # .reshape(-1)
                class_labels.append(int(cls.item()))
        return binary_masks, class_labels, class_names


    def rgbd_to_pointcloud(self, depth_image):
        height, width = depth_image.shape
        pointcloud = np.zeros((height, width, 3))

        # Create a meshgrid of coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Compute the X and Y coordinates
        Z = depth_image
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # Combine into a single point cloud
        pointcloud[..., 0] = X
        pointcloud[..., 1] = Y
        pointcloud[..., 2] = Z

        pointcloud = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

        downsampled_pointcloud = self.downsample_pointcloud(pointcloud)

        return downsampled_pointcloud


    def create_colored_pointcloud(self):
        global source_frame, target_frame
        if self.rgb_image is None or self.depth_image is None:
            return
        
        pointcloud = self.rgbd_to_pointcloud(self.depth_image)
        binary_masks, class_labels, class_names = self.get_segmentation_masks_and_classes(self.rgb_image)

        target_class_mask = np.zeros(self.depth_image.shape, dtype=bool)

        for mask, cls in zip(binary_masks, class_labels):
            if cls == 0:
                target_class_mask = np.logical_or(target_class_mask.flatten(), mask)
        
        pointcloud = pointcloud[target_class_mask.flatten()]

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        pointcloud_msg = pc2.create_cloud_xyz32(header, pointcloud)
        pointcloud_msg.fields.append(pc2.PointField(name='rgb', offset=12, datatype=7, count=1))
        pointcloud_msg.is_dense = False

       # Transform the point cloud to the target frame
        transformed_pointcloud_msg = self.transform_pointcloud(pointcloud_msg, 'rgbd_camera', 'map')
        
        # publish the transformed pointcloud
        self.pointcloud_pub.publish(transformed_pointcloud_msg)

    def transform_pointcloud(self, pointcloud, source_frame, target_frame):
        try:
            self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.0))
            (trans, rot) = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))

            transform_matrix = tf.transformations.quaternion_matrix(rot)
            transform_matrix[0:3, 3] = trans
            
            
            # could downsample points here

            # Convert point cloud to a NumPy array
            points = np.array(list(pc2.read_points(pointcloud, field_names=("x", "y", "z"), skip_nans=True)))
            
            # Add a fourth dimension of 1s to the points for homogeneous coordinates
            ones = np.ones((points.shape[0], 1))
            points_homogeneous = np.hstack((points, ones))

            # Apply the transformation matrix
            transformed_points_homogeneous = np.dot(points_homogeneous, transform_matrix.T)

            # Remove the homogeneous coordinate
            transformed_points = transformed_points_homogeneous[:, :3]

            # Create the transformed point cloud
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = target_frame
            transformed_pointcloud = pc2.create_cloud_xyz32(header, transformed_points)
            transformed_pointcloud.fields.append(pc2.PointField(name='rgb', offset=12, datatype=7, count=1))
            transformed_pointcloud.is_dense = False
            return transformed_pointcloud
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Error transforming point cloud: {e}")
            return pointcloud
        

    def downsample_pointcloud(self, points, voxel_size=0.25):
        # Create Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Downsample the point cloud
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        downsampled_points = np.asarray(downsampled_pcd.points)

        # Interpolate the downsampled point cloud back to the original resolution
        original_xy = points[:, :2]
        downsampled_z = griddata(downsampled_points[:, :2], downsampled_points[:, 2], original_xy, method='nearest')

        # Construct the interpolated point cloud
        interpolated_pointcloud = np.hstack((original_xy, downsampled_z.reshape(-1, 1)))

        return interpolated_pointcloud

if __name__ == '__main__':
    rospy.init_node('segmentation_pointcloud_node', anonymous=True)
    spc = SegmentationPointCloud()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        spc.create_colored_pointcloud()
        rate.sleep()
