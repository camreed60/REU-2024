#!/usr/bin/env python3

import rospy
import math
import random
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import time
import matplotlib.pyplot as plt

class PoseListener:
    def __init__(self):
        self.vehicleX = 0.0
        self.vehicleY = 0.0
        self.vehicleZ = 0.0

        rospy.Subscriber('/state_estimation', Odometry, self.pose_callback)

    def pose_callback(self, msg):
        self.vehicleX = msg.pose.pose.position.x
        self.vehicleY = msg.pose.pose.position.y
        self.vehicleZ = msg.pose.pose.position.z

    def get_vehicle_position(self):
        return self.vehicleX, self.vehicleY, self.vehicleZ

    def spin(self):
        rospy.spin()  # Keeps the node running until terminated

# A function that generates a best path from initial X, initial Y to 
# final X, final Y using RRT*.
# Returns a list of ordered X, Y points. The points are the nodes on
# the best path
def best_path(initial_x, initial_y, final_x, final_y, boundary_coordinates):
    # Initialize nodes with the initial position
    nodes = {(initial_x, initial_y): {'parent': None, 'cost': 0}}
    goal_radius = 1.0  # Radius for goal proximity
    min_step_size = 2.0  # Minimum step size for each iteration
    step_size = 4.0  # Step size for each iteration
    search_radius = 8.0  # Search radius for nearby nodes

    # Function to get the cost of a node
    def cost(node):
        return nodes[node]['cost'] if node in nodes else float('inf')

    # Function to set the parent of a node
    def set_parent(node, parent):
        nodes[node]['parent'] = parent

    # Function to set the cost of a node
    def set_cost(node, cost_value):
        nodes[node]['cost'] = cost_value

    # Function to get the parent of a node
    def get_parent(node):
        return nodes[node]['parent'] if node in nodes else None

    # Function to calculate the distance between two nodes
    def distance(node1, node2):
        return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

    # Function to reconstruct the path from the goal to the start
    def reconstruct_path(goal):
        path = []
        current_node = goal
        while current_node:
            path.append((current_node[0], current_node[1]))
            current_node = get_parent(current_node)
        path.reverse()
        return path

    def sample_free():
        # Use boundary box to sample points
        # Boundary_coordinates is a list of 4 pairs of coordinates
        min_x = min(coordinate[0] for coordinate in boundary_coordinates)
        max_x = max(coordinate[0] for coordinate in boundary_coordinates)
        min_y = min(coordinate[1] for coordinate in boundary_coordinates)
        max_y = max(coordinate[1] for coordinate in boundary_coordinates)
        
        return (random.uniform(min_x, max_x), random.uniform(min_y, max_y))

    # Function to find the nearest node to a given node
    def nearest(node):
        return min(nodes, key=lambda n: distance(n, node))

    # Function to steer from one node to another
    def steer(from_node, to_node):
        if distance(from_node, to_node) >  min_step_size and distance(from_node, to_node) < step_size:
            return to_node
        theta = math.atan2(to_node[1] - from_node[1], to_node[0] - from_node[0])
        if distance(from_node, to_node) > min_step_size:
            return (from_node[0] + step_size * math.cos(theta), from_node[1] + step_size * math.sin(theta))
        else:
            return (from_node[0] + min_step_size * math.cos(theta), from_node[1] + min_step_size * math.sin(theta))

    # Function to find nodes near a new node
    def near_nodes(new_node):
        return [node for node in nodes if distance(node, new_node) <= search_radius]

    # Start a timer
    start_time = time.time()
    goal_found = False
    while not goal_found:
        random_point = sample_free()  # Sample a random point
        nearest_node = nearest(random_point)  # Find the nearest node to the random point
        new_node = steer(nearest_node, random_point)  # Steer towards the random point
        new_cost = cost(nearest_node) + distance(nearest_node, new_node)  # Calculate the cost of the new node
        
        # If the new node is not in nodes or its cost is less than the current cost
        if new_node not in nodes or new_cost < cost(new_node):
            nodes[new_node] = {'parent': nearest_node, 'cost': new_cost}  # Add the new node to nodes

            # For each node near the new node
            for near_node in near_nodes(new_node):
                if near_node == nearest_node:
                    continue
                new_near_cost = new_cost + distance(new_node, near_node)  # Calculate the cost of the near node
                # If the new near cost is less than the current cost of the near node
                if new_near_cost < cost(near_node):
                    set_parent(near_node, new_node)  # Set the parent of the near node to the new node
                    set_cost(near_node, new_near_cost)  # Set the cost of the near node to the new near cost

            # If the new node is within the goal radius
            if distance(new_node, (final_x, final_y)) < goal_radius:
                nodes[(final_x, final_y)] = {'parent': new_node, 'cost': new_cost + distance(new_node, (final_x, final_y))}  # Add the goal to nodes
                goal_found = True  # Set goal found to True
                break
        
        # Get current time in relation to the start time
        current_time = time.time() - start_time
        # If the current time is greater than 60 seconds, break out
        # This ensures that the loop does not run indefinitely
        if current_time > 60:
            break

    # If the goal was not found
    if not goal_found:
        # Find the nearest node to the goal
        nearest_node = nearest((final_x, final_y))
        # Add the final point to nodes 
        nodes[(final_x, final_y)] = {'parent': nearest_node, 'cost': cost(nearest_node) + distance(nearest_node, (final_x, final_y))}
    
    # Add an additional 1,000 nodes to improve the tree
    for _ in range(1000):
        random_point = sample_free()  # Sample a random point
        nearest_node = nearest(random_point)  # Find the nearest node to the random point
        new_node = steer(nearest_node, random_point)  # Steer towards the random point
        new_cost = cost(nearest_node) + distance(nearest_node, new_node)  # Calculate the cost of the new node
        
        # If the new node is not in nodes or its cost is less than the current cost
        if new_node not in nodes or new_cost < cost(new_node):
            nodes[new_node] = {'parent': nearest_node, 'cost': new_cost}  # Add the new node to nodes

            # For each node near the new node
            for near_node in near_nodes(new_node):
                if near_node == nearest_node:
                    continue
                new_near_cost = new_cost + distance(new_node, near_node)  # Calculate the cost of the near node
                # If the new near cost is less than the current cost of the near node
                if new_near_cost < cost(near_node):
                    set_parent(near_node, new_node)  # Set the parent of the near node to the new node
                    set_cost(near_node, new_near_cost)  # Set the cost of the near node to the new near cost

    # Return the reconstructed path from the goal to the start
    return reconstruct_path((final_x, final_y))

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

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

    poseListener = PoseListener()
    # Set a two second pause before this line is executed
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()

    # Pause for 2 seconds
    time.sleep(2)
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
    path = best_path(vehicleX, vehicleY, finalX, finalY, [(0, 0), (0, 100), (100, 100), (100, 0)])
    rospy.loginfo("A path has been generated.")
    # Display solution using Matplotlib
    plot_solution(path)
    # Publish the waypoints in the path
    navigate_path(path, way_pub, rate, poseListener)
    
    # If the agent is not close to the final coordinates,
    # move back to the starting positon
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
    path_size = len(path) 
    # Check if the vehicle is not close to the final waypoint
    if (abs(vehicleX - path[path_size - 1][0]) > 1.0 and
        abs(vehicleY - path[path_size - 1][1]) > 1.0):
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
