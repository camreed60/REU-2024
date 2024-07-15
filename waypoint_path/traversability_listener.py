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
        self.terrain_points_list = []
        self.traversability_values = None
        self.travs_x = None
        self.travs_y = None

        # Subscribe to the /stitched_pointcloud topic
        rospy.Subscriber("/stitched_pointcloud", PointCloud2, self.callback_classes)

        # Subscribe to the /stitched_terrain_map topic
        rospy.Subscriber("/stitched_terrain_map", PointCloud2, self.callback_terrain)
    
    def controller(self):
        travs_result = self.build_traversability_map()
        map_q1 = travs_result[0]
        map_q2 = travs_result[1]
        map_q3 = travs_result[2]
        map_q4 = travs_result[3]
        terrain_result = self.build_terrain_map()
        if (terrain_result is not False):
            optimize_result = self.optimize_traversability_map(map_q1, map_q2, map_q3, map_q4,
                                                               terrain_result[0], terrain_result[1],
                                                               terrain_result[2], terrain_result[3])
            return optimize_result[0], optimize_result[1], optimize_result[2], optimize_result[3]
        else:
            return map_q1, map_q2, map_q3, map_q4
    
    def callback_classes(self, point_cloud):
        self.cloud_points = pc2.read_points(point_cloud, skip_nans=True)
        self.points_list = list(self.cloud_points)
    
    def callback_terrain(self, point_cloud):
        self.terrain_cloud_points = pc2.read_points(point_cloud, skip_nans=True)
        self.terrain_points_list = list(self.terrain_cloud_points)

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
            result = self.initialize_empty_quadrants(min_x, max_x, min_y, max_y, scale_value, 0.0)
            map_q1 = result[0]
            map_q2 = result[1]
            map_q3 = result[2]
            map_q4 = result[3]

            # Plotting
            for point in points_list:
                color_tuple = self.extract_color_tuple(point)
                traversability_value = self.convert_colors_to_traversability_value(color_tuple)
                traversability_values.append(traversability_value)
            self.traversability_values = traversability_values
            self.travs_x = x_coords
            self.travs_y = y_coords
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

    # Function to initialize empty quadrant maps
    def initialize_empty_quadrants(self, min_x, max_x, min_y, max_y, scale_value, initial_value):
        if (min_y > 0 and max_y > 0 and min_x > 0 and max_x > 0):
            map_q1 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x) + 1), initial_value)
            map_q2 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x) + 1), initial_value)
            map_q3 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x) + 1), initial_value)
            map_q4 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x) + 1), initial_value)
        elif (min_y > 0 and max_y > 0 and max_x <= 0):
            map_q1 = np.full((scale_value * int(max_y) + 1, scale_value * int(abs(min_x) + 1)), initial_value)
            map_q2 = np.full((scale_value * int(max_y) + 1, scale_value * int(abs(min_x) + 1)), initial_value)
            map_q3 = np.full((scale_value * int(max_y) + 1, scale_value * int(abs(min_x) + 1)), initial_value)
            map_q4 = np.full((scale_value * int(max_y) + 1, scale_value * int(abs(min_x) + 1)), initial_value)
        elif (max_y <= 0 and min_x > 0 and max_x > 0):
            map_q1 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x) + 1), initial_value)
            map_q2 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x) + 1), initial_value)
            map_q3 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x) + 1), initial_value)
            map_q4 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x) + 1), initial_value)
        elif (min_y > 0 and max_y > 0 and min_x <= 0 and max_x > 0):
            map_q1 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
            map_q2 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
            map_q3 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
            map_q4 = np.full((scale_value * int(max_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
        elif (min_y <= 0 and max_y > 0 and min_x > 0 and max_x > 0):
            map_q1 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x) + 1), initial_value)
            map_q2 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x) + 1), initial_value)
            map_q3 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x) + 1), initial_value)
            map_q4 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x) + 1), initial_value)
        elif (max_y > 0 and max_x > 0):
            map_q1 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
            map_q2 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
            map_q3 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
            map_q4 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(max_x - min_x) + 1), initial_value)
        elif max_y <= 0 and max_x > 0:
            map_q1 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), initial_value)
            map_q2 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), initial_value)
            map_q3 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), initial_value)
            map_q4 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(max_x - min_x) + 1), initial_value)
        elif max_y <= 0 and max_x <= 0:
            map_q1 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), initial_value)
            map_q2 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), initial_value)
            map_q3 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), initial_value)
            map_q4 = np.full((scale_value * int(abs(min_y) + 1), scale_value * int(abs(min_x) + 1)), initial_value)
        elif max_y > 0 and max_x <= 0:
            map_q1 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), initial_value)
            map_q2 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), initial_value)
            map_q3 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), initial_value)
            map_q4 = np.full((scale_value * int(max_y - min_y) + 1, scale_value * int(abs(min_x))), initial_value)
        return map_q1, map_q2, map_q3, map_q4

    def extract_color_tuple(self, point):
        r = int(point[3] * 255)
        g = int(point[4] * 255)
        b = int(point[5] * 255)
        return (r, g, b)

    def convert_colors_to_traversability_value(self, color_tuple):
        # Define the color mappings with tolerance
        # Simulation:
        
        color_map = {
            (0, 255, 0): 0.2,       
            (255, 255, 0): 1.0,
            (255, 0, 0): 0.0                
        } 
        '''
        # Define colors for each class
        color_map = {
            (255, 255, 0): 0.2,   # yellow : grass    
            (255, 128, 0): 0.0,    # Orange : rock 
            (0, 255, 0): 1.0,   # green : rocky-trail   
            (0, 0, 255): 0.05,  # blue : roots 
            (255, 0, 0): 0.5,  # red: rough-trail
            (0, 255, 255): 0.0,  # cyan : structure 
            (150, 75, 0): 0.0,  # brown : tree-trunk
            (128, 0, 255): 0.0  # Purple : vegetation 
        }'''

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
                box_min = np.array([x_value, y_value])
                box_max = np.array([x_value + 1, y_value + 1])
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

    # Function to build terrain map
    def build_terrain_map(self):
        try:
            points_list = self.terrain_points_list
            scale_value = self.scale_value
            x_coords = [point[0] for point in points_list]
            y_coords = [point[1] for point in points_list]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            # Create a KDTree for the points
            points_array = np.array(points_list)
            kdtree = KDTree(points_array[:, :2])

            # Initialize empty arrays
            result = self.initialize_empty_quadrants(min_x, max_x, min_y, max_y, scale_value, 3.5)
            map_q1 = result[0]
            map_q2 = result[1]
            map_q3 = result[2]
            map_q4 = result[3]

            # Process each quadrant
            def process_quadrant_terrain(map_q, x_multiplier, y_multiplier):
                # Iterate through each element in the map
                for (x, y), element in np.ndenumerate(map_q):
                    # Get a list of intensity from within each block of the 2D array
                    x_value = (x * x_multiplier) / self.scale_value
                    y_value = (y * y_multiplier) / self.scale_value
                    box_min = np.array([x_value, y_value])
                    box_max = np.array([x_value + 1, y_value + 1])
                    indices = kdtree.query_ball_point((box_min + box_max) / 2, np.linalg.norm(box_max - box_min) / 2)
                    relevant_points = points_array[indices]
                    intensity_list = [point[3] for point in relevant_points]
                    # Find which value shows up the most in each block of the terrain map
                    if intensity_list:
                        # Find the average of intensity list
                        if (len(intensity_list) > 0):
                            most_common_intensity = sum(intensity_list) / len(intensity_list)
                        else:
                            most_common_intensity = 3.5
                        if (0 <= y < map_q.shape[0] and 0 <= x < map_q.shape[1]):
                            # Set map_q[y, x] to most common intensity
                            map_q[y, x] = most_common_intensity
            
            # Start timer to construct traversability map
            start_time = time.time()
            try:
                print("Scaling terrain map...")
                
                process_quadrant_terrain(map_q1, 1, 1)
                process_quadrant_terrain(map_q2, -1, 1)
                process_quadrant_terrain(map_q3, -1, -1)
                process_quadrant_terrain(map_q4, 1, -1)

                current_time = time.time() - start_time
                print("It took", current_time, "seconds to scale the terrain map.")
            except Exception as e:
                print("There was a problem:", e)
                current_time = time.time() - start_time
                print("Process ran for", current_time, "seconds.")

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
            # The first quadrant is displayed in the top-right
            axs[0, 1].imshow(map_q1, cmap='viridis', origin='lower')
            axs[0, 1].set_title('Quadrant 1')
            
            # The second quadrant is flipped horizontally and displayed in the top-left
            axs[0, 0].imshow(np.fliplr(map_q2), cmap='viridis', origin='lower')
            axs[0, 0].set_title('Quadrant 2')
            
            # The third quadrant is flipped both horizontally and vertically and displayed in the bottom-left
            axs[1, 0].imshow(np.flip(map_q3), cmap='viridis', origin='lower')
            axs[1, 0].set_title('Quadrant 3')
            
            # The fourth quadrant is flipped vertically, displayed in the bottom-right
            axs[1, 1].imshow(np.flipud(map_q4), cmap='viridis', origin='lower')
            axs[1, 1].set_title('Quadrant 4')

            return map_q1, map_q2, map_q3, map_q4
        except:
            return False

    def optimize_traversability_map(self, travs_map_q1, travs_map_q2, travs_map_q3, travs_map_q4,
                                    terrain_map_q1, terrain_map_q2, terrain_map_q3, terrain_map_q4):
        
        print("Optimizing the traversability map based on the results from the terrain map...")

        # Function to apply optimization rules
        def optimize_quadrant(travs_map, terrain_map):
            for i in range(travs_map.shape[0]):
                for j in range(travs_map.shape[1]):
                    try:
                        if terrain_map[i, j] == 3.5:
                            continue  # Disregard elements with value 3.5 in terrain_map
                        # Corrects traversable areas that are being incorrectly flagged as non-traversable
                        elif terrain_map[i, j] <= 0.3:
                            if travs_map[i, j] == 0:
                                travs_map[i, j] = 0.5
                        # Corrects non-traversable areas that are being incorrectly flagged as traversable
                        elif terrain_map[i, j] >= 1.0:
                            if travs_map[i, j] == 1:
                                travs_map[i, j] = 0
                    except:
                        continue
            return travs_map

        # Apply optimization rules for each quadrant
        travs_map_q1 = optimize_quadrant(travs_map_q1, terrain_map_q1)
        travs_map_q2 = optimize_quadrant(travs_map_q2, terrain_map_q2)
        travs_map_q3 = optimize_quadrant(travs_map_q3, terrain_map_q3)
        travs_map_q4 = optimize_quadrant(travs_map_q4, terrain_map_q4)

        # Return the updated traversability maps
        return travs_map_q1, travs_map_q2, travs_map_q3, travs_map_q4

    # Generate an empty map in the case where a traversability map cannot be constructed
    def generate_empty_map(self, width, height):
        return np.random.uniform(low=1, high=1, size=(width,height))