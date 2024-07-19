#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import io

class PointCloudSubscriber:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('pointcloud_subscriber', anonymous=True)
        
        # Subscribe to the stitched_pointcloud topic
        self.subscriber = rospy.Subscriber('/traversability_map', PointCloud2, self.callback)
        
        # Define the file path
        self.file_path = 'arboretum_2.bin'
        
        rospy.loginfo("PointCloudSubscriber initialized and listening to /traversability_map topic")
    
    def callback(self, msg):
        rospy.loginfo("Received point cloud data, saving to file")
        try:
            # Create a buffer for serialization
            buff = io.BytesIO()
            
            # Serialize the PointCloud2 message into the buffer
            msg.serialize(buff)
            
            # Write the serialized data to a binary file
            with open(self.file_path, 'wb') as file:
                file.write(buff.getvalue())
                
            rospy.loginfo("Successfully saved point cloud data to %s", self.file_path)
        except Exception as e:
            rospy.logerr("Failed to save point cloud data: %s", str(e))

if __name__ == '__main__':
    try:
        PointCloudSubscriber()
        # Keep the node running
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

