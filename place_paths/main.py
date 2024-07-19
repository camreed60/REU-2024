#!/usr/bin/env python3

import pickle
import rospy
import math
from geometry_msgs.msg import PointStamped
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pose_listener import PoseListener
from metrics import MetricCollection

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

    # Pause for 2 seconds
    time.sleep(2)

    # Initialize the pose listener
    poseListener = PoseListener()
    # Set a two second pause before this line is executed
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()

    # Pause for 2 seconds
    time.sleep(2)

    # Get values from data object
    with open('waypoint_path.dat', 'rb') as f:
        pickle_data = pickle.load(f)
    
    finalX = pickle_data['finalX']
    finalY = pickle_data['finalY']
    scale = pickle_data['scale']
    quad1 = pickle_data['quad1']
    quad2 = pickle_data['quad2']
    quad3 = pickle_data['quad3']
    quad4 = pickle_data['quad4']
    path = pickle_data['path']

    # Display path
    plot_solution(path, scale)

    # Start metrics collection
    metrics = MetricCollection(quad1, quad2, quad3, quad4, scale)
    # Pause for 5 seconds
    time.sleep(5)
    metrics.start_timer()
    metrics.start_distance_traversed()
    metrics.start_time_on_trail()

    # Publish the waypoints in the path
    navigate_path(path, way_pub, rate, poseListener, scale)

    # If the agent is not close to the final coordinates,
    # move back to the starting positon
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
    # Check if the vehicle is not close to the final waypoint
    if (abs(vehicleX - finalX) > 1.0 or
        abs(vehicleY - finalY) > 1.0):
        print("The final point was not reached. Returning to the starting positon.")
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
   
    # End metrics collection
    metrics.end_timer()
    metrics.end_time_on_trail()
    total_distance_traversed = metrics.end_distance_traversed()
    print("Total time:", metrics.total_time)
    print("Percent time on trail:", metrics.percent_time_on_trail)
    print("Total distance traversed:", total_distance_traversed)

# Function that publishes waypoints sequentially from the start position to the goal position
def navigate_path(path, way_pub, rate, poseListener, scale):
    # Iterate through path in order
    for point in path:
        # Start a timer
        start_time = time.time()
        while not rospy.is_shutdown():
            vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
            print("Current vehicle position: ({}, {}, {})".format(vehicleX, vehicleY, vehicleZ))

            # Publish waypoint
            waypoint = PointStamped()
            waypoint.header.stamp = rospy.Time.now()
            waypoint.header.frame_id = "map"
            waypoint.point.x = point[0] / scale
            waypoint.point.y = point[1] / scale
            waypoint.point.z = vehicleZ
            way_pub.publish(waypoint)
            print("Published waypoint: {}".format(waypoint))
            
            # Check if the vehicle is near the current waypoint
            if (abs(vehicleX - (point[0] / scale)) < 1.0 and
                abs(vehicleY - (point[1] / scale)) < 1.0 ):
                break  # Move to the next waypoint

            # Get current time in relation to the start time
            current_time = time.time() - start_time
            # If the current time is greater than 15 seconds, move to the next waypoint
            # This ensures that the loop does not run indefinitely
            if current_time > 15:
                print("Waypoint not reached, moving to the next one...")
                break
            
            rate.sleep()

def semantic_based_data(quad1, quad2, quad3, quad4, way_pub, rate, poseListener, scale, path, traversListener):
    # Semantic Based
    traverse_time_list = []
    percent_time_list = []
    distance_traversed_list = []
    for i in range(0, 10):
        # Get the current position of the vehicle
        vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
        if (i == 0 or i == 2 or i== 4 or i== 6 or i == 8 or i == 10):
            finalX = 26
            finalY = 10
        else:
            finalX = 8
            finalY = -12
        # Start metrics collection
        metrics = MetricCollection(quad1, quad2, quad3, quad4, scale)
        # Pause for 5 seconds
        time.sleep(5)
        metrics.start_timer()
        metrics.start_distance_traversed()
        metrics.start_time_on_trail()
        navigate_path(path, way_pub, rate, poseListener, scale)
        metrics.end_timer()
        metrics.end_time_on_trail()
        total_distance_traversed = metrics.end_distance_traversed()
        print("Total time:", metrics.total_time)
        print("Percent time on trail:", metrics.percent_time_on_trail)
        print("Total distance traversed:", total_distance_traversed)
        traverse_time_list.append(metrics.total_time)
        percent_time_list.append(metrics.percent_time_on_trail)
        distance_traversed_list.append(total_distance_traversed)   

    print("Traverse Times List")
    for i in traverse_time_list:
        print(i)
    print("Percent List")
    for i in percent_time_list:
        print(i)
    print("Distance List")
    for i in distance_traversed_list:
        print(i)  

def geometric_based_data(way_pub, poseListener, quad1, quad2, quad3, quad4, scale):
    traverse_time_list = []
    percent_time_list = []
    distance_traversed_list = []
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
    
    for i in range(0, 10):
        if (i == 0 or i == 2 or i== 4 or i== 6 or i == 8 or i == 10):
            finalX = 26
            finalY = 10
        else:
            finalX = 26
            finalY = 10
        # Pause for 5 seconds
        time.sleep(5)
        # Start metrics collection
        metrics = MetricCollection(quad1, quad2, quad3, quad4, scale)
        # Pause for 5 seconds
        time.sleep(5)
        metrics.start_timer()
        metrics.start_distance_traversed()
        metrics.start_time_on_trail()
        loop = True
        while loop:
            waypoint = PointStamped()
            waypoint.header.stamp = rospy.Time.now()
            waypoint.header.frame_id = "map"
            waypoint.point.x = finalX
            waypoint.point.y = finalY
            waypoint.point.z = vehicleZ
            way_pub.publish(waypoint)
            rospy.loginfo("Published waypoint: {}".format(waypoint))
            vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
            # Check if the vehicle is near the current waypoint
            if (abs(vehicleX - (finalX)) < 1.0 and
                abs(vehicleY - (finalY)) < 1.0 ):
                break
        metrics.end_timer()
        metrics.end_time_on_trail()
        total_distance_traversed = metrics.end_distance_traversed()
        print(i)
        print("Total time:", metrics.total_time)
        print("Percent time on trail:", metrics.percent_time_on_trail)
        print("Total distance traversed:", total_distance_traversed)
        traverse_time_list.append(metrics.total_time)
        percent_time_list.append(metrics.percent_time_on_trail)
        distance_traversed_list.append(total_distance_traversed)  
        
    print("Traverse Times List")
    for i in traverse_time_list:
        print(i)
    print("Percent List")
    for i in percent_time_list:
        print(i)
    print("Distance List")
    for i in distance_traversed_list:
        print(i)  

if __name__=='__main__':
    try:
        random_waypoint_publisher()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted")
    except Exception as e:
        rospy.logerr("An error occurred: {}".format(str(e)))