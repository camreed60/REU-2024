#!/usr/bin/env python3

import rospy
import subprocess
import time
import signal

def wait_for_roscore():
    while True:
        try:
            rospy.get_master().getPid()
            return
        except:
            print("Waiting for roscore...")
            time.sleep(1)

def main():
    # Start roscore if it's not already running
    try:
        rospy.get_master().getPid()
    except:
        print("Starting roscore...")
        subprocess.Popen(["roscore"])
    
    wait_for_roscore()

    # Start the rosbag play process
    bag_file = "/home/wvuirl/ros_environment/autonomous_exploration_ws/src/datasets/pathway_bags/ros_bag_pathways_*.bag"  # Replace with your bag file path
    rosbag_process = subprocess.Popen(["rosbag", "play", "--pause", bag_file], stdin=subprocess.PIPE)

    print("Rosbag is paused. Press Enter to start playback...")
    input()  # Wait for user to press Enter

    # Send a space character to the rosbag play process to unpause
    rosbag_process.stdin.write(b' ')
    rosbag_process.stdin.flush()

    # Wait for the rosbag play process to finish
    try:
        rosbag_process.wait()
    except KeyboardInterrupt:
        print("Interrupted by user, stopping rosbag...")
        rosbag_process.send_signal(signal.SIGINT)
        rosbag_process.wait()

if __name__ == "__main__":
    main()