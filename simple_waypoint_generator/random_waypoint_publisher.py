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
FINAL_Z = 0.005

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

# A function that generates a best path from initial X, initial Y, initial Z to 
# final X, final Y, final Z using RRT*.
# Returns a dictionary of ordered X, Y, and Z points. The points are the nodes on
# the best path
def best_path(initial_x, initial_y, initial_z):
    # Initialize variables
    nodes = {(initial_x, initial_y, initial_z): {'parent': None, 'cost': 0}}  # Start with the initial point
    max_iter = 1000  # Maximum number of iterations
    goal_radius = 1.0  # Distance threshold to the final point
    
    # Helper functions for managing nodes
    def cost(node):
        return nodes[node]['cost'] if node in nodes else float('inf')
    
    def set_parent(node, parent):
        if node in nodes:
            nodes[node]['parent'] = parent
    
    def set_cost(node, cost_value):
        if node in nodes:
            nodes[node]['cost'] = cost_value
    
    def get_parent(node):
        return nodes[node]['parent'] if node in nodes else None
    
    def reconstruct_path(goal):
        # Reconstruct the path from nodes to goal using parent pointers
        path = []
        current_node = goal
        
        while current_node:
            path.append({'x': current_node[0], 'y': current_node[1], 'z': current_node[2]})
            current_node = get_parent(current_node)
        
        path.reverse()  # Reverse path to start from initial point
        return path
    
    # RRT* algorithm
    goal_found = False
    for _ in range(max_iter):
    
        # Generate a random point around the last added node
        last_added = list(nodes.keys())[-1]
        random_point = (random.uniform(last_added[0] - 1, last_added[0] + 1),
                        random.uniform(last_added[1] - 1, last_added[1] + 1),
                        random.uniform(last_added[2] - 1, last_added[2] + 1))
        
        # Find the nearest node in nodes to the random_point
        min_dist = float('inf')
        nearest_node = None
        
        for node in nodes:
            dist = math.sqrt((node[0] - random_point[0])**2 + 
                             (node[1] - random_point[1])**2 + 
                             (node[2] - random_point[2])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        # Attempt to connect nearest_node to random_point
        if nearest_node:
            # Calculate cost to reach the random_point through nearest_node
            new_cost = cost(nearest_node) + min_dist
            
            if (abs(random_point[0] - FINAL_X) < 0.5) and (abs(random_point[1] - FINAL_Y) < 0.5) and (abs(random_point[2] - FINAL_Z) < 0.5):
                # Add the final point to nodes and break
                nodes[(FINAL_X, FINAL_Y, FINAL_Z)] = {'parent': nearest_node, 'cost': new_cost + min_dist}
                goal_found = True
                break
            else:
                # Add the random_point as a new node
                nodes[random_point] = {'parent': nearest_node, 'cost': new_cost}
                
                # Rewire the tree to optimize costs
                for node in nodes:
                    if node != nearest_node:
                        dist_to_new_node = math.sqrt((node[0] - random_point[0])**2 + 
                                                     (node[1] - random_point[1])**2 + 
                                                     (node[2] - random_point[2])**2)
                        if dist_to_new_node < goal_radius:
                            new_node_cost = new_cost + dist_to_new_node
                            if new_node_cost < cost(node):
                                set_parent(node, random_point)
                                set_cost(node, new_node_cost)
    
    if not goal_found:
        # Add the final point to nodes 
        nodes[(FINAL_X, FINAL_Y, FINAL_Z)] = {'parent': nearest_node, 'cost': new_cost + min_dist}
        
    # Construct the path from nodes
    path = reconstruct_path((FINAL_X, FINAL_Y, FINAL_Z))
    return path

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
    path = best_path(vehicleX, vehicleY, vehicleZ)
    rospy.loginfo("Best path generated: {}".format(path))
    
    # Iterate through path in order
    for point in path:
        while not rospy.is_shutdown():
            vehicleX, vehicleY, vehicleZ = poseListener.get_vehicle_position()
            rospy.loginfo("Current vehicle position: ({}, {}, {})".format(vehicleX, vehicleY, vehicleZ))

            waypoint = PointStamped()
            waypoint.header.stamp = rospy.Time.now()
            waypoint.header.frame_id = "map"
            waypoint.point.x = point['x']
            waypoint.point.y = point['y']
            waypoint.point.z = point['z']

            way_pub.publish(waypoint)
            rospy.loginfo("Published waypoint: {}".format(waypoint))
            
            # Check if the vehicle is near the current waypoint
            if (abs(vehicleX - point['x']) < 0.5 and
                abs(vehicleY - point['y']) < 0.5 ):
                break  # Move to the next waypoint
            
            rate.sleep()

if __name__=='__main__':
    try:
        random_waypoint_publisher()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted")
    except Exception as e:
        rospy.logerr("An error occurred: {}".format(str(e)))
