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

# Camera parameters
fx, fy = 205.47, 205.47  # Focal lengths
cx, cy = 320.5, 180.5  # Principal points

# Model weights
model_path = "/home/wvuirl/ros_environment/autonomous_exploration_ws/src/filter_segmented_classes/src/walkway_v1.pt"

class SegmentationPointCloud:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO(model_path)
        self.pointcloud_pub = rospy.Publisher('/segmented_pointcloud', PointCloud2, queue_size=10)
        depth_sub = Subscriber('/rgbd_camera/depth/image', Image)
        rgb_sub = Subscriber('/rgbd_camera/color/image', Image)
        ts = ApproximateTimeSynchronizer([depth_sub, rgb_sub], 10, 0.05) # .1 not .05
        ts.registerCallback(self.synchronized_callback)

        # Define colors for each class
        self.class_colors = {
            0: (255, 0, 0),    # Red : grass
            1: (0, 255, 0),    # Green : gravel
            2: (0, 0, 255),    # Blue : mulch
            3: (255, 255, 0),  # Yellow : obstacle
            4: (255, 0, 255),  # Magenta : parking lot
            5: (0, 255, 255),  # Cyan : sidewalk
            6: (255, 128, 0),  # Orange : unused
            7: (128, 0, 255),  # Purple : vegetation
        }

    def synchronized_callback(self, depth_msg, rgb_msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        self.create_colored_pointcloud()

    def get_segmentation_masks_and_classes(self, img):
        results = self.model.predict(img)
        class_names = self.model.names
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
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        Z = depth_image
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    def create_colored_pointcloud(self):
        downsampled_depth = cv2.resize(self.depth_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        downsampled_fx, downsampled_fy = fx * 0.5, fy * 0.5
        downsampled_cx, downsampled_cy = cx * 0.5, cy * 0.5
        pointcloud = self.rgbd_to_pointcloud(downsampled_depth, downsampled_fx, downsampled_fy, downsampled_cx, downsampled_cy)

        initial_point_count = pointcloud.shape[0]

        binary_masks, class_labels, class_names = self.get_segmentation_masks_and_classes(self.rgb_image)

        resized_masks = [cv2.resize(mask.reshape(self.rgb_image.shape[:2]), 
                                    (downsampled_depth.shape[1], downsampled_depth.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST) for mask in binary_masks]

        colored_pointcloud = np.zeros((pointcloud.shape[0], 6))  # XYZ + RGB
        colored_pointcloud[:, :3] = pointcloud

        # Create a mask for all classified points
        total_mask = np.zeros(resized_masks[0].shape, dtype=bool)

        # Assign colors based on class
        for mask, class_label in zip(resized_masks, class_labels):
            mask_bool = mask.astype(bool)
            color = self.class_colors.get(class_label, (128, 128, 128))
            colored_pointcloud[mask_bool.flatten(), 3:] = color
            # Convert 0-255 RGB to 0-1 float
            color_float = np.array(color, dtype=np.float32) / 255.0
            colored_pointcloud[mask_bool.flatten(), 3:] = color_float
            total_mask |= mask_bool

        # Keep only classified points
        colored_pointcloud = colored_pointcloud[total_mask.flatten()]

        # Create and publish PointCloud2 message
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'rgbd_camera'
        fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1)]
        pointcloud_msg = pc2.create_cloud(header, fields, colored_pointcloud)
        self.pointcloud_pub.publish(pointcloud_msg)

if __name__ == '__main__':
    rospy.init_node('segmentation_pointcloud_node', anonymous=True)
    spc = SegmentationPointCloud()
    rospy.spin()