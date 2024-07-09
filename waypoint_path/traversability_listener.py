#!/usr/bin/env python3

import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt
import io

# TODO: In the future, this class will be used to
# construct a 2D Traversability Map and will then 
# return it to the waypoint path node for use with 
# the advanced RRT Star Path Planner.

class TraversabilityListener:
    def __init__(self, scale_value):
        self.points_list = []
        self.scale_value = int(scale_value)

        # Subscribe to the /stitched_pointcloud topic
        rospy.Subscriber("/stitched_pointcloud", PointCloud2, self.callback)
        
    def callback(self, point_cloud):
        self.cloud_points = pc2.read_points(point_cloud, skip_nans=True)
        self.points_list = list(self.cloud_points)

    def build_traversability_map(self):
        scale_value = self.scale_value

        points_list = self.points_list
        x_coords = [point[0] for point in points_list]
        y_coords = [point[1] for point in points_list]


        if len(x_coords) > 0 and len(y_coords) > 0:
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            traversability_values = []
            
            # Initialize empty maps for each quadrant
            if (max_y > 0 and max_x > 0):
                map_q1 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), 0.0)
                map_q2 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), 0.0)
                map_q3 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), 0.0)
                map_q4 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), 0.0)
            elif max_y <= 0 and max_x > 0:
                map_q1 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), 0.0)
                map_q2 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), 0.0)
                map_q3 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), 0.0)
                map_q4 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), 0.0)
            elif max_y <= 0 and max_x <= 0:
                map_q1 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), 0.0)
                map_q2 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), 0.0)
                map_q3 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), 0.0)
                map_q4 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), 0.0)
            elif max_y > 0 and max_x <= 0:
                map_q1 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), 0.0)
                map_q2 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), 0.0)
                map_q3 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), 0.0)
                map_q4 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), 0.0)

            for point in points_list:
                x, y = point[:2]
                y_scale = scale_value * y 
                x_scale = scale_value * x
                y_value = int(y_scale)
                x_value = int(x_scale)
                color_tuple = self.extract_color_tuple(point)
                traversability_value = self.convert_colors_to_traversability_value(color_tuple)
                traversability_values.append(traversability_value)
                # First quadrant
                if x >= 0 and y >= 0:
                    map_q1[abs(y_value), abs(x_value)] = traversability_value
                # Second quadrant (x values are made positive)
                elif x < 0 and y >= 0:
                    map_q2[abs(y_value), abs(x_value)] = traversability_value
                # Third quadrant (x and y values are made positive)
                elif x < 0 and y < 0:
                    map_q3[abs(y_value), abs(x_value)] = traversability_value
                # Fourth quadrant (y values are made positive)
                elif x >= 0 and y < 0:
                    map_q4[abs(y_value), abs(x_value)] = traversability_value
            
            # Plotting
            plt.figure(figsize=(10, 8))
            plt.scatter(x_coords, y_coords, c=traversability_values, cmap='viridis', s=10, alpha=0.8)
            plt.colorbar(label='Traversability Value')
            plt.title('Traversability Map')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            plt.show()

            return map_q1, map_q2, map_q3, map_q4

    def extract_color_tuple(self, point):
        r = int(point[3] * 255)
        g = int(point[4] * 255)
        b = int(point[5] * 255)
        return (r, g, b)

    def convert_colors_to_traversability_value(self, color_tuple):
        # Define the color mappings with tolerance
        color_map = {
            (255, 0, 0): 0.8,           # Red: Grass 
            (0, 255, 0): 0.5,           # Green: Gravel
            (0, 0, 255): 1.0,           # Blue: Mulch
            (255, 255, 0): 0.0,         # Yellow: Obstacle
            (255, 0, 255): 0.0,         # Magenta: Parking Lot
            (0, 255, 255): 1.0,         # Cyan: Path 
            (255, 128, 0): 0.0,         # Orange: Unused
            (128, 0, 255): 0.2          # Purple: Vegetation
        }

        # Define tolerance for color matching
        tolerance = 20

        # Check if the color tuple matches any color within the tolerance
        for key, value in color_map.items():
            if all(abs(color_tuple[i] - key[i]) <= tolerance for i in range(3)):
                return value

        # Default value for any unknown color
        return 0.0

    # Generate an empty map for testing purposes
    def generate_empty_map(self, width, height):
        return np.random.uniform(low=1, high=1, size=(width,height))
