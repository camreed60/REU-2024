#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
import cv2
from cv_bridge import CvBridge
import numpy as np

# declaring global variables
bridge = CvBridge()
camera_proj_matrix = None
segmented_image = None

def camera_info_callback(msg):
    global camera_proj_matrix
    camera_proj_matrix = np.reshape(msg.P, (3,4))

def segmented_image_callback(msg):
    global segmented_image
    segmented_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') #  or mono8

def point_cloud_callback(msg):
    global camera_proj_matrix, segmented_image

    if camera_proj_matrix is None or segmented_image is None:
        return
    
    # project point cloud to image plane
    points_list = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    points_3d = np.array([p[:3] for p in points_list], dtype=np.float32)
    points_3d = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
    points_2d = (points_3d @ camera_proj_matrix.T)[:, :2] / points_3d[:, 3:]
    points_2d = points_2d.astype(int)

    # filter point cloud based on segmentation labels
    filtered_points = []
    for point_3d, point_2d in zip(points_list, points_2d):
        x, y = point_2d
        if 0 <= x < segmented_image.shape[1] and 0 <= y < segmented_image.shape[0]:
            label = segmented_image[y, x]
            if label in desired_classes:    
                filtered_points.append(point_3d)

    # publish filtered point cloud
    filtered_cloud = point_cloud2.create_cloud(msg.header, msg.fields, filtered_points)
    filtered_cloud_pub.publish(filtered_cloud)


if __name__ == "__main__":
    rospy.init_node("point_cloud_filter")

    camera_info_sub = rospy.Subscriber("/rgbd_camera/color/image", Image, camera_info_callback)
    segmented_image_sub = rospy.Subscriber("/segmented_image", Image, segmented_image_callback)
    point_cloud_sub = rospy.Subscriber("/rgbd_camera/depth/points", PointCloud2, point_cloud_callback)
    filtered_cloud_pub = rospy.Publisher("/filtered_point_cloud", PointCloud2, queue_size=1)

    desired_classes = ['sidewalk', 'grass', 'obstacle']     # edit classes here

    rospy.spin()