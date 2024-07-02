#!/usr/bin/env python3

# this script generates a point cloud from a depth image and visualizes the points within the mask as pink, else black
import numpy as np
import open3d as o3d

# Define the mask
mask = np.zeros((480, 640), dtype=int)
mask[:240, :325] = 1 
mask_flat = mask.reshape(-1)

# Camera parameters
fx = 205.47  # Focal length in x direction (example value)
fy = 205.47  # Focal length in y direction (example value)
cx = 320.5  # Principal point x (example value)
cy = 180.5  # Principal point y (example value)

# Generate an example depth image
depth = np.random.uniform(0, 5, (480, 640))  # Example depth image

def rgbd_to_pointcloud(depth, fx, fy, cx, cy):
    """
    Convert RGB-D image to a point cloud.

    :param depth: Depth image (H, W)
    :param fx: Focal length in x direction
    :param fy: Focal length in y direction
    :param cx: Principal point x
    :param cy: Principal point y
    :return: Point cloud as (N, 3) array where N is number of points and columns are (X, Y, Z)
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

def visualize_point_cloud(points, colors):
    """
    Visualize the point cloud with specified colors.

    :param points: Point cloud data (N, 3)
    :param colors: Colors for the points (N, 3)
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])

pointcloud = rgbd_to_pointcloud(depth, fx, fy, cx, cy)

# Create the color array based on the mask
colors = np.zeros_like(pointcloud)  # Initialize with black
colors[mask_flat == 1] = [1.0, 0.0, 1.0]  # Set masked points to pink

print(pointcloud.shape)
print(colors.shape)

# Visualize the point cloud with the specified colors
visualize_point_cloud(pointcloud, colors)
