#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import cv2

# this script takes a depth image and generates a point cloud from it

def rgbd_to_pointcloud(depth, fx, fy, cx, cy):
    """
    Convert RGB-D image to a point cloud.

    :param rgb: RGB image (H, W, 3)
    :param depth: Depth image (H, W)
    :param fx: Focal length in x direction
    :param fy: Focal length in y direction
    :param cx: Principal point x
    :param cy: Principal point y
    :param baseline: Baseline distance between stereo cameras
    :return: Point cloud as (N, 6) array where N is number of points and columns are (X, Y, Z, R, G, B)
    """
    height, width = depth.shape
    pointcloud = np.zeros((height, width, 3))

    for v in range(height):
        for u in range(width):
            Z = depth[v, u]
            if Z > 0:  # Valid depth value
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                pointcloud[v, u] = [X, Y, Z]

    return pointcloud.reshape(-1, 3)


def visualize_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([point_cloud])


# Camera parameters
fx = 322.28  # Focal length in x direction (example value) # 205.47 in sim # rs 322.28
fy = 322.28 # Focal length in y direction (example value) # 205.47 in sim # rs 322.28
cx = 320.82  # Principal point x (example value) # 320.5 in sim # rs 320.82
cy = 178.77  # Principal point y (example value) # 180.5 in sim # rs 178.77

depth_path = '/home/wvuirl/ros_environment/autonomous_exploration_ws/src/filter_segmented_classes/src/depth_image.png'
# Assuming rgb and depth are your input images
#rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Example RGB image
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

pointcloud = rgbd_to_pointcloud(depth, fx, fy, cx, cy)

print(pointcloud.shape)
visualize_point_cloud(pointcloud)

