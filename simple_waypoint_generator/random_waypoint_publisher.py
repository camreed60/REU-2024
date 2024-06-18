#!/usr/bin/env python3

import rospy
import math
import random
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import time

# 7.092166900634766, 2.7193148136138916, 0.7565169930458069
FINAL_X = 12
FINAL_Y = 78

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
    max_iter = 10000  # Maximum number of iterations
    goal_radius = 1.0  # Radius for goal proximity
    min_step_size = 2.0  # Minimum step size for each iteration
    step_size = 4.0  # Step size for each iteration
    search_radius = 4.0  # Search radius for nearby nodes

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
        if distance(from_node, to_node) >  min_step_size:
            return (from_node[0] + min_step_size * math.cos(theta), from_node[1] + min_step_size * math.sin(theta))
        else:
            return (from_node[0] + step_size * math.cos(theta), from_node[1] + step_size * math.sin(theta))

    # Function to find nodes near a new node
    def near_nodes(new_node):
        return [node for node in nodes if distance(node, new_node) <= search_radius]

    goal_found = False
    #for _ in range(max_iter):
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

    # If the goal was not found
    if not goal_found:
        # Add the final point to nodes 
        nodes[(final_x, final_y)] = {'parent': nearest_node, 'cost': new_cost + distance(new_node, (final_x, final_y))}

    # Return the reconstructed path from the goal to the start
    return reconstruct_path((final_x, final_y))

def random_waypoint_publisher():
    # Initialize publisher node
    rospy.init_node('random_waypoint_publisher', anonymous=True)

    # To publish to waypoint topic
    way_pub = rospy.Publisher('/way_point', PointStamped, queue_size=1)
    rate = rospy.Rate(1)

    poseListener = PoseListener()
    # Set a two second pause before this line is executed
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()

    # Pause for 2 seconds
    time.sleep(2)
    vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
    path = best_path(vehicleX, vehicleY, FINAL_X, FINAL_Y, [(0, 0), (0, 100), (100, 100), (100, 0)])
    rospy.loginfo("Best path generated: {}".format(path))
    
    # Iterate through path in order
    for point in path:
        while not rospy.is_shutdown():
            vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
            rospy.loginfo("Current vehicle position: ({}, {}, {})".format(vehicleX, vehicleY, vehicleZ))

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
            
            rate.sleep()

if __name__=='__main__':
    try:
        random_waypoint_publisher()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted")
    except Exception as e:
        rospy.logerr("An error occurred: {}".format(str(e)))
