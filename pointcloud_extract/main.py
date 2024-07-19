#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import io

def read_pointcloud_file_and_publish(file_path, publisher):
    rate = rospy.Rate(1)  # Adjust the publishing rate as needed (1 Hz in this example)
    
    while not rospy.is_shutdown():
        try:
            # Read the binary file
            with open(file_path, 'rb') as file:
                serialized_data = file.read()
            
            # Create a PointCloud2 message
            point_cloud_msg = PointCloud2()
            
            # Create a BytesIO buffer and write the serialized data to it
            buff = io.BytesIO(serialized_data)
            
            # Deserialize the buffer into the PointCloud2 message
            point_cloud_msg.deserialize(buff.getvalue())
            
            # Publish the PointCloud2 message
            publisher.publish(point_cloud_msg)
            print("Published point cloud data from file to traversability_map topic")

        except IOError:
            rospy.logerr("Error reading file: %s", file_path)

        rate.sleep()

def listener():
    # Initialize the ROS node
    rospy.init_node('pointcloud_publisher', anonymous=True)
    
    # Create a publisher for the stitched_pointcloud topic
    publisher = rospy.Publisher('/traversability_map', PointCloud2, queue_size=10)
    
    # Read and continuously publish the contents of pointcloud.bin
    read_pointcloud_file_and_publish('arboretum_2.bin', publisher)
    
    # Keep the script running
    rospy.spin()

if __name__ == '__main__':
    listener()