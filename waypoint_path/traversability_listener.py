#!/usr/bin/env python3

import rospy
import numpy as np

# TODO: In the future, this class will be used to get a 2D Traversability Map from another
# node (subscribe to a publisher) and will then return it to the waypoint path node for use
# with the advanced RRT Star Path Planner.

class TraversabilityListener:
    def __init__(self):
        pass

    def generate_empty_map(self):
        width, height = 500, 500
        traversability_map = np.random.uniform(low=1, high=1, size=(width,height))
        return traversability_map