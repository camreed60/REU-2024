#!/usr/bin/env python3
import rospy
import numpy as np
from ultralytics import YOLO
import cv2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros
from geometry_msgs.msg import TransformStamped
import open3d as o3d

# Camera parameters
fx, fy = 260.68658447265625, 260.68658447265625  # Focal lengths
cx, cy = 317.3788757324219, 182.31150817871094  # Principal points

# Model weights
model_path = "/home/wvuirl/ros_environment/autonomous_exploration_ws/src/traversability_mapping/src/2000imgs_weights.pt"

class SegmentationPointCloud:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO(model_path)
        self.pointcloud_pub = rospy.Publisher('/segmented_pointcloud', PointCloud2, queue_size=10)
        depth_sub = Subscriber('/zed2i/zed_node/depth/depth_registered', Image) # /rgbd_camera/depth/image
        rgb_sub = Subscriber('/zed2i/zed_node/rgb_raw/image_raw_color', Image) # /rgbd_camera/color/image
        ts = ApproximateTimeSynchronizer([depth_sub, rgb_sub], 10, 0.25)  # Reduced tolerance
        ts.registerCallback(self.synchronized_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.height_threshold = 1.0

        # Define colors for each class
        self.class_colors = {
            0: (255, 255, 0),    # yellow : grass    
            1: (255, 128, 0),    # Orange : rock 
            2: (0, 255, 0),    # green : rocky-trail   
            3: (0, 0, 255),  # blue : roots 
            4: (255, 0, 0),  # red: rough-trail
            5: (0, 255, 255),  # cyan : structure 
            6: (150, 75, 0),  # brown : tree-trunk
            7: (128, 0, 255),  # Purple : vegetation 
        }

    def get_robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0))
            return np.array([transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get robot position: {e}")
            return None
        
    def synchronized_callback(self, depth_msg, rgb_msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        self.create_colored_pointcloud(depth_msg.header.stamp)

    def get_segmentation_masks_and_classes(self, img):
        results = self.model.predict(img)
        class_names = self.model.names
        binary_masks, class_labels = [], []
        if results[0].masks is not None:
            masks = results[0].masks.data
            classes = results[0].boxes.cls
            #non_vegetation = classes != 7
            #masks = masks[non_vegetation]
            #classes = classes[non_vegetation]
            masks_np = masks.cpu().numpy().astype(np.uint8)
            masks_np = np.array([cv2.resize(mask, (self.depth_image.shape[1], self.depth_image.shape[0])) for mask in masks_np])
            binary_masks = masks_np.reshape(masks_np.shape[0], -1)
            class_labels = classes.cpu().numpy().astype(int)
        return binary_masks, class_labels, class_names

    def rgbd_to_pointcloud(self, depth_image, fx, fy, cx, cy):
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        Z = depth_image
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    def create_colored_pointcloud(self, timestamp):
        pointcloud = self.rgbd_to_pointcloud(self.depth_image, fx, fy, cx, cy)

        binary_masks, class_labels, class_names = self.get_segmentation_masks_and_classes(self.rgb_image)

        colored_pointcloud = np.zeros((pointcloud.shape[0], 6))  # XYZ + RGB
        colored_pointcloud[:, :3] = pointcloud

        total_mask = np.zeros(self.depth_image.shape, dtype=bool)

        for mask, class_label in zip(binary_masks, class_labels):
            mask_bool = mask.reshape(self.depth_image.shape).astype(bool)
            color = self.class_colors.get(class_label, (128, 128, 128))
            color_float = np.array(color, dtype=np.float32) / 255.0
            colored_pointcloud[mask_bool.flatten(), 3:] = color_float
            total_mask |= mask_bool

        colored_pointcloud = colored_pointcloud[total_mask.flatten()]

        # Transform pointcloud to account for robot movement
        try:
            transform = self.tf_buffer.lookup_transform('map', 'zed_link', rospy.Time(0))
            colored_pointcloud[:, :3] = self.transform_pointcloud(colored_pointcloud[:, :3], transform)

            robot_position = self.get_robot_position()
            if robot_position is not None:
                height_threshold = self.height_threshold
                height_mask = colored_pointcloud[:, 2] <= (robot_position[2] + height_threshold)
                colored_pointcloud = colored_pointcloud[height_mask]
                
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to transform pointcloud: {e}")

        header = std_msgs.msg.Header()
        header.stamp = timestamp
        header.frame_id = 'map'
        fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
                pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
                pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1)]
        pointcloud_msg = pc2.create_cloud(header, fields, colored_pointcloud)
        self.pointcloud_pub.publish(pointcloud_msg)

    def transform_pointcloud(self, points, transform):
        # Convert transform to 4x4 matrix
        t = transform.transform.translation
        r = transform.transform.rotation
        T = np.array([
            [1-2*(r.y**2+r.z**2), 2*(r.x*r.y-r.z*r.w), 2*(r.x*r.z+r.y*r.w), t.x],
            [2*(r.x*r.y+r.z*r.w), 1-2*(r.x**2+r.z**2), 2*(r.y*r.z-r.x*r.w), t.y],
            [2*(r.x*r.z-r.y*r.w), 2*(r.y*r.z+r.x*r.w), 1-2*(r.x**2+r.y**2), t.z],
            [0, 0, 0, 1]
        ])

        # Create a rotation matrix to rotate 90 degrees around the Y-axis
        rotation_z = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]

        ])
        rotation_y = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        combined_rotation = np.dot(rotation_y, rotation_z)
        combined_transform = np.dot(T, combined_rotation)

        # Add homogeneous coordinate
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        
        # Transform points
        points_transformed = np.dot(combined_transform, points_h.T).T[:, :3]
        return points_transformed

if __name__ == '__main__':
    rospy.init_node('segmentation_pointcloud_node', anonymous=True)
    spc = SegmentationPointCloud()
    rospy.spin()