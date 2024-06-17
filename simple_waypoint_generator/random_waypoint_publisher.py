#!/usr/bin/env python3

import rospy
import random
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2 
import sensor_msgs.point_cloud2 as pc2

selected_point[0] = 0
selected_point[1] = 0
selected_point[2] = 0

def sensor_scan_callback(scan_data):
    global selected_point
    points = []
    for point in pc2.read_points(scan_data, field_names=("x", "y", "z"), skip_nans=True):
        points.append((point[0], point[1], point[2]))
    # Select a random point with Z value between -1 and 1
    while not selected_point:
        random_point = random.choice(points)
        if -1 <= random_point[2] <= 1:
            selected_point = random_point

def random_waypoint_publisher():
    # Initialize publisher node
    rospy.init_node('random_waypoint_publisher', anonymous = True)
    # Subscribe to sensor scan topic
    rospy.Subscriber('/sensor_scan', PointCloud2, sensor_scan_callback)  # Subscribe to the sensor_scan topic

    # To publish to waypoint topic
    way_pub = rospy.Publisher('/way_point', PointStamped)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        waypoint = PointStamped()
        waypoint.header.stamp = rospy.Time.now()
        waypoint.header.frame_id = "map"
        waypoint.point.x = selected_point[0]
        waypoint.point.y = selected_point[1]
        waypoint.point.z = selected_point[2]

        way_pub.publish(waypoint)
        rospy.loginfo("Published random waypoint: {}".format(waypoint))

        selected_point = None  # Reset selected_point
        rate.sleep()

if __name__=='__main__':
    try:
        random_waypoint_publisher()
    except:
        pass