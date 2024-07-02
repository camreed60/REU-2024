#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

class ImageSubscriber:
    def __init__(self, output_folder, frame_rate, num_images):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/zed2i/zed_node/rgb_raw/image_raw_color", Image, self.callback)
        self.output_folder = output_folder
        self.frame_rate = frame_rate
        self.num_images = num_images
        self.total_frames = int(430 * frame_rate)  # Assuming 1000 seconds of video
        self.frame_interval = self.total_frames // num_images
        self.frame_count = 0
        self.saved_count = 0

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def callback(self, data):
        if self.saved_count >= self.num_images:
            rospy.signal_shutdown("Captured required number of images")
            return

        if self.frame_count % self.frame_interval == 0:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)
                return

            image_filename = os.path.join(self.output_folder, f"arboretum2_{self.saved_count:04d}.jpg")
            cv2.imwrite(image_filename, cv_image)
            self.saved_count += 1
            rospy.loginfo(f"Saved image {self.saved_count}/{self.num_images}")

        self.frame_count += 1

def main():
    output_folder = "/home/wvuirl/ros_environment/autonomous_exploration_ws/src/dataset_imagery"  # Change this to your desired output folder
    frame_rate = 15  # Frames per second of the video
    num_images = 1000  # Number of images you want to capture

    rospy.init_node('image_subscriber', anonymous=True)
    image_subscriber = ImageSubscriber(output_folder, frame_rate, num_images)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()