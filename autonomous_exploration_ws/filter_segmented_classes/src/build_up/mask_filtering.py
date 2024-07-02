#!/usr/bin/env python3

# this script takes a depth image, generates the pointcloud from it, and then filters the pointclouds based on masks
import numpy as np
import open3d as o3d


mask = np.zeros((480, 640), dtype=int)
mask[:240, :325] = 1 
mask_flat = mask.reshape(-1)

fx = 205.47  # Focal length in x direction (example value)
fy = 205.47  # Focal length in y direction (example value)
cx = 320.5  # Principal point x (example value)
cy = 180.5  # Principal point y (example value)

# Assuming rgb and depth are your input images
depth = np.random.uniform(0, 5, (480, 640))  # Example depth image


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


pointcloud = rgbd_to_pointcloud(depth, fx, fy, cx, cy)

# filter the points where the mask is 1
filtered_points = pointcloud[mask_flat == 1]

print(filtered_points.shape)
print(pointcloud.shape)
visualize_point_cloud(pointcloud)
visualize_point_cloud(filtered_points)

