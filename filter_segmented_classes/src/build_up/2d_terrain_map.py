import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
import matplotlib.patches as patches
import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2

class TerrainMapper:
    def __init__(self):
        rospy.init_node('terrain_mapper', anonymous=True)
        
        self.map_size = 50  # meters
        self.resolution = 0.1  # meters
        self.grid_size = int(self.map_size / self.resolution)
        
        self.robot_position = np.array([0.0, 0.0])
        self.map_origin = np.array([0.0, 0.0])
        
        self.terrain_grid = np.full((self.grid_size, self.grid_size), -1, dtype=int)
        
        self.class_colors = {
            0: (255, 0, 0),    # Red : grass
            1: (0, 255, 0),    # Green : gravel
            2: (0, 0, 255),    # Blue : mulch
            3: (255, 255, 0),  # Yellow : obstacle
            4: (255, 0, 255),  # Magenta : parking lot
            5: (0, 255, 255),  # Cyan : sidewalk
            6: (255, 128, 0),  # Orange : unused
            7: (128, 0, 255),  # Purple : vegetation
        }
        
        self.class_names = ['grass', 'gravel', 'mulch', 'obstacle', 'parking lot', 'sidewalk', 'unused', 'vegetation']
        
        color_list = [(128/255, 128/255, 128/255)]  # Add gray for unknown
        color_list += [tuple(c/255 for c in color) for color in self.class_colors.values()]
        self.cmap = colors.ListedColormap(color_list)
        self.bounds = range(len(color_list) + 1)
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.im = self.ax.imshow(self.terrain_grid, cmap=self.cmap, norm=self.norm,
                                 extent=[0, self.map_size, 0, self.map_size])
        self.colorbar = self.fig.colorbar(self.im, ticks=range(len(color_list)))
        self.colorbar.set_ticklabels(['unknown'] + self.class_names)
        
        self.ax.set_title('Terrain Classification Map')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        
        self.robot_marker = self.ax.add_patch(patches.Circle((self.map_size/2, self.map_size/2), 0.5, color='black', zorder=5))
        
        rospy.Subscriber("/segmented_pointcloud", PointCloud2, self.pointcloud_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        self.points_processed = 0
        self.update_interval = 1000  # Update plot every 1000 points

    def world_to_grid(self, x, y):
        grid_x = int((x - self.map_origin[0]) / self.resolution)
        grid_y = int((y - self.map_origin[1]) / self.resolution)
        return grid_x, grid_y

    def pointcloud_callback(self, data):
        print(f"Received PointCloud2 message with {data.width * data.height} points")
        for point in pc2.read_points(data, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=True):
            x, y, z, r, g, b = point
            
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)
            
            terrain_class = self.get_terrain_class(r, g, b)
            
            grid_x, grid_y = self.world_to_grid(x, y)
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.terrain_grid[grid_y, grid_x] = terrain_class
            
            self.points_processed += 1
            if self.points_processed % self.update_interval == 0:
                print(f"Processed {self.points_processed} points. Updating plot...")
                self.update_plot_manual()

        print(f"Finished processing PointCloud2 message. Total points processed: {self.points_processed}")
        self.update_plot_manual()

    def get_terrain_class(self, r, g, b):
        input_color = (r, g, b)
        for class_id, color in self.class_colors.items():
            if input_color == color:
                return class_id + 1  # Shift by 1 because -1 is for unknown
        print(f"Unknown color: ({r}, {g}, {b})")
        return 0  # Unknown class (was -1, now 0 in the color map)

    def odom_callback(self, msg):
        self.robot_position[0] = msg.pose.pose.position.x
        self.robot_position[1] = msg.pose.pose.position.y
        print(f"Robot position updated: {self.robot_position}")

    def update_plot_manual(self):
        # Update map origin to keep robot centered
        self.map_origin = self.robot_position - np.array([self.map_size/2, self.map_size/2])
        
        # Update image extent
        extent = [self.map_origin[0], self.map_origin[0] + self.map_size,
                  self.map_origin[1], self.map_origin[1] + self.map_size]
        self.im.set_extent(extent)
        
        # Update image data
        self.im.set_array(self.terrain_grid)
        
        # Update robot marker position (should always be at the center)
        self.robot_marker.center = (self.map_size/2, self.map_size/2)
        
        # Update axis limits
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self):
        plt.show(block=False)
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            plt.pause(0.1)
            rate.sleep()

if __name__ == '__main__':
    try:
        mapper = TerrainMapper()
        mapper.run()
    except rospy.ROSInterruptException:
        pass