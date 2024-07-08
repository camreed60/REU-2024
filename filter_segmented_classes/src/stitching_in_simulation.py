#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d


class PointCloudStitcher:
    def __init__(self):
        self.pointcloud_sub = rospy.Subscriber('/segmented_pointcloud', PointCloud2, self.pointcloud_callback)
        self.stitched_pointcloud_pub = rospy.Publisher('/stitched_pointcloud', PointCloud2, queue_size=10)
        self.global_pointcloud = None
        self.voxel_size = .25 # 10cm voxel size, adjust as needed

    def pointcloud_callback(self, pointcloud_msg):
        local_pointcloud = self.pointcloud2_to_array(pointcloud_msg)
        self.update_global_pointcloud(local_pointcloud)
        self.publish_stitched_pointcloud(pointcloud_msg.header)

    def update_global_pointcloud(self, new_pointcloud):
        if self.global_pointcloud is None:
            self.global_pointcloud = new_pointcloud
        else:
            # Combine with existing global pointcloud
            self.global_pointcloud = np.vstack((self.global_pointcloud, new_pointcloud))
            
        # Downsample the combined pointcloud
        self.global_pointcloud = self.voxel_downsample(self.global_pointcloud)

    def voxel_downsample(self, pointcloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:])
        downsampled_pcd = pcd.voxel_down_sample(self.voxel_size)
        return np.hstack([np.asarray(downsampled_pcd.points), np.asarray(downsampled_pcd.colors)])

    def publish_stitched_pointcloud(self, header):
        header.frame_id = 'map'  # Ensure the stitched pointcloud is in the map frame
        fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('r', 12, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('g', 16, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('b', 20, pc2.PointField.FLOAT32, 1)]
        pointcloud_msg = pc2.create_cloud(header, fields, self.global_pointcloud)
        self.stitched_pointcloud_pub.publish(pointcloud_msg)

    @staticmethod
    def pointcloud2_to_array(cloud_msg):
        points_list = []
        for point in pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z", "r", "g", "b")):
            points_list.append(point)
        return np.array(points_list, dtype=np.float32)

if __name__ == '__main__':
    rospy.init_node('pointcloud_stitcher_node', anonymous=True)
    stitcher = PointCloudStitcher()
    rospy.spin()
