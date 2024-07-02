#!/usr/bin/env python3

# this script generates a point cloud from a depth image and segmented rgb image. each class will get a unique color
import numpy as np
import open3d as o3d
from ultralytics import YOLO
import cv2

# Camera parameters
fx = 322.28  # Focal length in x direction (example value) # 205.47 in sim # rs 322.28
fy = 322.28 # Focal length in y direction (example value) # 205.47 in sim # rs 322.28
cx = 320.82  # Principal point x (example value) # 320.5 in sim # rs 320.82
cy = 178.77  # Principal point y (example value) # 180.5 in sim # rs 178.77

# load depth image, rgb image for segmentation, model weights
depth_path = "/home/wvuirl/ros_environment/autonomous_exploration_ws/src/filter_segmented_classes/src/depth_image1.png"
img_path = "/home/wvuirl/ros_environment/autonomous_exploration_ws/src/filter_segmented_classes/src/color_image1.jpg"
model_path = "/home/wvuirl/ros_environment/autonomous_exploration_ws/src/filter_segmented_classes/src/walkway_v1.pt"

def get_segmentation_masks_and_classes(image_path, model_path):
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Load image
    img = cv2.imread(image_path)

    # Perform inference
    results = model.predict(img)

    # Extract class names from the model
    class_names = model.names

    # Extract segmentation masks and class labels
    binary_masks = []
    class_labels = []
    if results[0].masks is not None:
        masks = results[0].masks.data
        for mask, cls in zip(masks, results[0].boxes.cls):
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8).reshape(-1)
            binary_masks.append(mask_np)
            class_labels.append(int(cls.item()))

    return binary_masks, class_labels, class_names


def rgbd_to_pointcloud(depth_image, fx, fy, cx, cy):
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

# load depth image
depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# get pointcloud
pointcloud = rgbd_to_pointcloud(depth_image, fx, fy, cx, cy)

# get segmentation masks and class labels
binary_masks, class_labels, class_names = get_segmentation_masks_and_classes(img_path, model_path)

# Define different colors for each class
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

print(pointcloud.shape)
print(colors.shape)

# Visualize the point cloud with the specified colors
visualize_point_cloud(pointcloud, colors)
