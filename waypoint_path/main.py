#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import PointStamped
import time
import matplotlib.pyplot as plt
from pose_listener import PoseListener
from path_planner import RRTStarPathPlanner
from advanced_path_planner import AdvancedRRTStarPathPlanner
from traversability_listener import TraversabilityListener

# Function that calculates the distance between two points in a 2D space
def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# Function that displays plot of RRT* graph solution
def plot_solution(path):
    # Extract x and y coordinates from the path
    x_coords = [node[0] for node in path]
    y_coords = [node[1] for node in path]

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

    # Set X and Y scale
    plt.xlim(0, 150)
    plt.ylim(0, 150)

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

    # Initialize the pose listener
    poseListener = PoseListener()
    # Set a two second pause before this line is executed
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()

    # Pause for 2 seconds
    time.sleep(2)
    # Get the current position of the vehicle
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
    # Initialize the traversability listener
    traversListener = TraversabilityListener()
    # Generate a blank traversability map
    # In the future, this will instead get the actual one
    traversability_map = traversListener.generate_empty_map()
    # Initialize the path planner
    planner = RRTStarPathPlanner(vehicleX, vehicleY, finalX, finalY, [(0, 0), (0, 100), (100, 100), (100, 0)])
    # Initialize the advanced path planner
    advanced_planner = AdvancedRRTStarPathPlanner(vehicleX, vehicleY, finalX, finalY, [(0, 0), (0, 100), (100, 100), (100, 0)], traversability_map, 1000)
    # Plan the path
    path = planner.plan_path()
    rospy.loginfo("A path has been generated.")
    # Display solution using Matplotlib
    plot_solution(path)
    # Publish the waypoints in the path
    navigate_path(path, way_pub, rate, poseListener)
    
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
            distance = calculate_distance(vehicleX, vehicleY, point[0], point[1])
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
        navigate_path(path_segment, way_pub, rate, poseListener)

# Function that publishes waypoints sequentially from the start position to the goal position
def navigate_path(path, way_pub, rate, poseListener):
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
            waypoint.point.x = point[0]
            waypoint.point.y = point[1]
            waypoint.point.z = vehicleZ
            way_pub.publish(waypoint)
            rospy.loginfo("Published waypoint: {}".format(waypoint))
            
            # Check if the vehicle is near the current waypoint
            if (abs(vehicleX - point[0]) < 1.0 and
                abs(vehicleY - point[1]) < 1.0 ):
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
