#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import PointStamped
import time
import numpy as np
import matplotlib.pyplot as plt
from pose_listener import PoseListener
from path_planner import RRTStarPathPlanner
from advanced_path_planner import AdvancedRRTStarPathPlanner
from traversability_listener import TraversabilityListener

# Function to display four quadrant traversability map
def display_four_quadrant_traversability_map(travs_quad1, travs_quad2, travs_quad3, travs_quad4):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # The first quadrant is displayed in the top-right
    axs[0, 1].imshow(travs_quad1, cmap='viridis', origin='lower')
    axs[0, 1].set_title('Quadrant 1')
    
    # The second quadrant is flipped horizontally and displayed in the top-left
    axs[0, 0].imshow(np.fliplr(travs_quad2), cmap='viridis', origin='lower')
    axs[0, 0].set_title('Quadrant 2')
    
    # The third quadrant is flipped both horizontally and vertically and displayed in the bottom-left
    axs[1, 0].imshow(np.flip(travs_quad3), cmap='viridis', origin='lower')
    axs[1, 0].set_title('Quadrant 3')
    
    # The fourth quadrant is flipped vertically, displayed in the bottom-right
    axs[1, 1].imshow(np.flipud(travs_quad4), cmap='viridis', origin='lower')
    axs[1, 1].set_title('Quadrant 4')
    
    # Set a common title, xlabel, and ylabel for the figure
    fig.suptitle('Traversability Map')
    fig.supxlabel('X')
    fig.supylabel('Y')
    
    plt.tight_layout()
    plt.show()

# (Not Completely Functional): Function to display four quadrant traversability map with path overlayed on it
def final_display_four_quadrant_traversability_map(travs_quad1, travs_quad2, travs_quad3, travs_quad4, path):
    # Get coordinates from final path in first quadrant
    path_quad1 = [(x, y) for (x, y) in path if x >= 0 and y >= 0]

    # Get coordinates from final path in second quadrant
    path_quad2 = [(x, y) for (x, y) in path if x < 0 and y >= 0]

    # Get coordinates from final path in third quadrant
    path_quad3 = [(x, y) for (x, y) in path if x < 0 and y < 0]

    # Get coordinates from final path in fourth quadrant
    path_quad4 = [(x, y) for (x, y) in path if x >= 0 and y < 0]
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # The first quadrant is displayed in the top-right
    axs[0, 1].imshow(travs_quad1, cmap='viridis', origin='lower')
    if (path_quad1 is not None):
        axs[0, 1].plot([x for x, y in path_quad1], [y for x, y in path_quad1], color='red', marker='o')
    axs[0, 1].set_title('Quadrant 1')
    
    # The second quadrant is flipped horizontally and displayed in the top-left
    axs[0, 0].imshow(np.fliplr(travs_quad2), cmap='viridis', origin='lower')
    if (path_quad2 is not None):
        axs[0, 0].plot([-x for x, y in path_quad2], [y for x, y in path_quad2], color='red', marker='o')
    axs[0, 0].set_title('Quadrant 2')
    
    # The third quadrant is flipped both horizontally and vertically and displayed in the bottom-left
    axs[1, 0].imshow(np.flip(travs_quad3), cmap='viridis', origin='lower')
    if (path_quad3 is not None):
        axs[1, 0].plot([-x for x, y in path_quad3], [-y for x, y in path_quad3], color='red', marker='o')
    axs[1, 0].set_title('Quadrant 3')
    
    # The fourth quadrant is flipped vertically, displayed in the bottom-right
    axs[1, 1].imshow(np.flipud(travs_quad4), cmap='viridis', origin='lower')
    if (path_quad4 is not None):
        axs[1, 1].plot([x for x, y in path_quad4], [-y for x, y in path_quad4], color='red', marker='o')
    axs[1, 1].set_title('Quadrant 4')
    
    # Set a common title, xlabel, and ylabel for the figure
    fig.suptitle('Traversability Map with Final Path')
    fig.supxlabel('X')
    fig.supylabel('Y')
    
    plt.tight_layout()
    plt.show()

# Graph overlay function
def graph_overlay(traversability_map, path, width, height):
    plt.imshow(traversability_map, cmap='viridis', origin='lower')
    plt.colorbar(label='Traversability')
    plt.title('Traversability Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    # Add path
    x_coords = [node[0] for node in path]
    y_coords = [node[1] for node in path]
    
    plt.plot(x_coords, y_coords, 'o-', color='blue')
    plt.show()

# Function that calculates the distance between two points in a 2D space
def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# Function that displays plot of RRT* graph solution
def plot_solution(path, scale):
    # Extract x and y coordinates from the path and scale them
    x_coords = [node[0] / scale for node in path]
    y_coords = [node[1] / scale for node in path]

    # Create a new figure
    plt.figure()

    # Plot the path
    plt.plot(x_coords, y_coords, 'o-', color='blue')

    # Plot start and end points
    plt.plot(x_coords[0], y_coords[0], 'go')  # Start in green
    plt.plot(x_coords[-1], y_coords[-1], 'ro')  # End in red

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Best Path')

    # Calculate the limits for the plot based on scaled coordinates
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # Add padding to the limits
    padding = 10
    plt.xlim(min_x - padding, max_x + padding)
    plt.ylim(min_y - padding, max_y + padding)

    # Show the plot
    plt.show()

def random_waypoint_publisher():
    # Initialize publisher node
    rospy.init_node('random_waypoint_publisher', anonymous=True)

    # To publish to waypoint topic
    way_pub = rospy.Publisher('/way_point', PointStamped, queue_size=1)
    rate = rospy.Rate(1)

    finalX = rospy.get_param('~finalX', 10.0)  # Default to 10.0 if parameter is not found
    finalY = rospy.get_param('~finalY', 10.0)  # Default to 10.0 if parameter is not found
    scale = int(rospy.get_param('~scale', 5.0))  # Default to 5.0 if parameter is not found

    # Initialize the pose listener
    poseListener = PoseListener()
    # Set a two second pause before this line is executed
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()

    # Pause for 2 seconds
    time.sleep(2)
    # Get the current position of the vehicle
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
    # Initialize the traversability listener
    traversListener = TraversabilityListener(scale)
    # Pause for 5 seconds
    time.sleep(5)
    # Generate a blank traversability map (Use in the case where one cannot be constructed)
    traversability_map = traversListener.generate_empty_map(100, 100)
    try:
        quad1, quad2, quad3, quad4 = traversListener.controller()
        display_four_quadrant_traversability_map(quad1, quad2, quad3, quad4)
    except:
        rospy.loginfo("Cannot find a stitched pointcloud. Using a blank traversability map.")
        quad1 = traversability_map
        quad2 = traversability_map
        quad3 = traversability_map
        quad4 = traversability_map
    
    # Initialize the path planner
    planner = RRTStarPathPlanner(vehicleX, vehicleY, finalX, finalY, [(0, 0), (0, 100), (100, 100), (100, 0)])
    # Initialize the advanced path planner
    advanced_planner = AdvancedRRTStarPathPlanner(vehicleX, vehicleY, finalX, finalY, 10000, quad1, quad2, quad3, quad4, scale)
    # Plan the path
    path = advanced_planner.plan_path()
    rospy.loginfo("A path has been generated.")
    # Display solution using Matplotlib
    plot_solution(path, scale)
    final_display_four_quadrant_traversability_map(quad1, quad2, quad3, quad4, path)
    # Publish the waypoints in the path
    navigate_path(path, way_pub, rate, poseListener, scale)
    
    # If the agent is not close to the final coordinates,
    # move back to the starting positon
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
    # Check if the vehicle is not close to the final waypoint
    if (abs(vehicleX - finalX) > 1.0 or
        abs(vehicleY - finalY) > 1.0):
        rospy.loginfo("The final point was not reached. Returning to the starting positon.")
        # Initialize an empty list to store distances
        distances = []
        # Determine which point in the path is closest to vehicle's current positon
        for point in path:
            # Compare how close the point is to the current position
            vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
            # Calculate distance between vehicle and current point in the path
            distance = calculate_distance(vehicleX, vehicleY, (point[0] / scale), (point[1] / scale))
            # Append the distance to the list
            distances.append(distance)
        # Get the index of the shortest distance in the array
        index = 0
        lowest = float('inf')
        counter = 0
        for values in distances:
            if values < lowest:
                index = counter
                lowest = values
            counter += 1
        # Split the path at the point where the vehicle is currently closest to
        path_segment = path[:index+1] 
        path_segment.reverse()
        # Publish the waypoints in the path back to the starting position
        navigate_path(path_segment, way_pub, rate, poseListener, scale)

# Function that publishes waypoints sequentially from the start position to the goal position
def navigate_path(path, way_pub, rate, poseListener, scale):
    # Iterate through path in order
    for point in path:
        # Start a timer
        start_time = time.time()
        while not rospy.is_shutdown():
            vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
            rospy.loginfo("Current vehicle position: ({}, {}, {})".format(vehicleX, vehicleY, vehicleZ))

            # Publish waypoint
            waypoint = PointStamped()
            waypoint.header.stamp = rospy.Time.now()
            waypoint.header.frame_id = "map"
            waypoint.point.x = point[0] / scale
            waypoint.point.y = point[1] / scale
            waypoint.point.z = vehicleZ
            way_pub.publish(waypoint)
            rospy.loginfo("Published waypoint: {}".format(waypoint))
            
            # Check if the vehicle is near the current waypoint
            if (abs(vehicleX - (point[0] / scale)) < 1.0 and
                abs(vehicleY - (point[1] / scale)) < 1.0 ):
                break  # Move to the next waypoint

            # Get current time in relation to the start time
            current_time = time.time() - start_time
            # If the current time is greater than 15 seconds, move to the next waypoint
            # This ensures that the loop does not run indefinitely
            if current_time > 15:
                rospy.loginfo("Waypoint not reached, moving to the next one...")
                break
            
            rate.sleep()

if __name__=='__main__':
    try:
        random_waypoint_publisher()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted")
    except Exception as e:
        rospy.logerr("An error occurred: {}".format(str(e)))