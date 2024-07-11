#!/usr/bin/env python3

import rospy
import time
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Constructs a 2D Traversability Map and then 
# returns it to the waypoint path node for use with 
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
            # These are the traversability maps to be passed to the planner
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
            
            # Plotting
            for point in points_list:
                color_tuple = self.extract_color_tuple(point)
                traversability_value = self.convert_colors_to_traversability_value(color_tuple)
                traversability_values.append(traversability_value)
            plt.figure(figsize=(10, 8))
            plt.scatter(x_coords, y_coords, c=traversability_values, cmap='viridis', s=10, alpha=0.8)
            plt.colorbar(label='Traversability Value')
            plt.title('Traversability Map')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            plt.show()

            # Build the traversability maps for the planner
            self.optimize_quadrants(map_q1, map_q2, map_q3, map_q4)

            return map_q1, map_q2, map_q3, map_q4

    def extract_color_tuple(self, point):
        r = int(point[3] * 255)
        g = int(point[4] * 255)
        b = int(point[5] * 255)
        return (r, g, b)

    def convert_colors_to_traversability_value(self, color_tuple):
        # Define the color mappings with tolerance
        color_map = {
            (0, 255, 0): 0.2,       
            (255, 255, 0): 1.0,
            (255, 0, 0): 0.0                
        }

        # Define tolerance for color matching
        tolerance = 20

        # Check if the color tuple matches any color within the tolerance
        for key, value in color_map.items():
            if all(abs(color_tuple[i] - key[i]) <= tolerance for i in range(3)):
                return value

        # Default value for any unknown color
        return 0.0
    
    # Optimize the quadrants to build the traversability maps for the planner
    def optimize_quadrants(self, map_q1, map_q2, map_q3, map_q4):
        # Create a KDTree for the points
        points_array = np.array(self.points_list)
        kdtree = KDTree(points_array[:, :2])

        # Process each quadrant
        def process_quadrant(map_q, x_multiplier, y_multiplier):
            # Iterate through each element in the map
            for (x, y), element in np.ndenumerate(map_q):
                # Get a list of colors from within each block of the 2D array
                x_value = (x * x_multiplier) / self.scale_value
                y_value = (y * y_multiplier) / self.scale_value
                box_min = np.array([x_value - 1, y_value - 1])
                box_max = np.array([x_value , y_value ])
                indices = kdtree.query_ball_point((box_min + box_max) / 2, np.linalg.norm(box_max - box_min) / 2)
                relevant_points = points_array[indices]
                color_list = [self.extract_color_tuple(point) for point in relevant_points]
                # Find which color shows up the most in each block of the traversability map
                if color_list:
                    most_common_color = max(set(color_list), key=color_list.count)
                    if (0 <= y < map_q.shape[0] and 0 <= x < map_q.shape[1]):
                        traversability_value = self.convert_colors_to_traversability_value(most_common_color)
                        map_q[y, x] = traversability_value

        # Start timer to construct traversability map
        start_time = time.time()
        try:
            print("Scaling traversability map for the path planner...")
            process_quadrant(map_q1, 1, 1)
            process_quadrant(map_q2, -1, 1)
            process_quadrant(map_q3, -1, -1)
            process_quadrant(map_q4, 1, -1)
            current_time = time.time() - start_time
            print("It took", current_time, "seconds to scale the traversability map for the planner.")
        except Exception as e:
            print("There was a problem:", e)
            current_time = time.time() - start_time
            print("Process ran for", current_time, "seconds.")

    # Generate an empty map in the case where a traversability map cannot be constructed
    def generate_empty_map(self, width, height):
        return np.random.uniform(low=1, high=1, size=(width,height))
