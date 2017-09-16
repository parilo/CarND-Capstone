#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number

def dist(a, b):
    """
    Distance between two points

    Args:
        a (geometry_msgs/Point): first point
        b (geometry_msgs/Point): second point
    """
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.current_waypoints_index = 0

        rospy.spin()

    def pose_cb(self, msg):

        # need to determine nearest waypoint and store it index
        # we assume that we going through waypoints continousely
        # and waypoints don't change, so we cache current waypoint and search
        # from current position onwards.

        # return if we at the end of the path
        if self.current_waypoints_index == len(self.base_waypoints.waypoints) - 1:
            return

        car_pos =  msg.pose.position
        car_pos.z = 0
        prev_dist = dist (
            car_pos,
            self.base_waypoints.waypoints[self.current_waypoints_index].pose.pose.position
        )
        for i in range(self.current_waypoints_index + 1, len(self.base_waypoints.waypoints)):
            next_dist = dist (
                car_pos,
                self.base_waypoints.waypoints[i].pose.pose.position
            )
            if next_dist > prev_dist:
                break
            self.current_waypoints_index = i
            prev_dist = next_dist

        final_waypoints = Lane()
        final_waypoints.header.frame_id = '/world'
        final_waypoints.header.stamp = rospy.Time(0)
        final_waypoints_end_index = min(self.current_waypoints_index + LOOKAHEAD_WPS + 1, len(self.base_waypoints.waypoints))
        final_waypoints.waypoints = self.base_waypoints.waypoints[self.current_waypoints_index + 2:final_waypoints_end_index]

        self.final_waypoints_pub.publish(final_waypoints)

    def waypoints_cb(self, waypoints):
        # possibly we need to handle situation of waypoints chaning
        # and reset self.current_waypoints_index to 0 in this case
        # here
        self.base_waypoints = waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance_along_waypoints(self, waypoints, wp1, wp2):
        """
        Calculate distance between two waypoints

        Args:
            waypoints (array of waypoits):
            wp1 (int): index of start waypoints
            wp2 (int): index of end waypoint

        Returns:
            float: distance along waypoints
        """
        dist = 0
        # dl = lambda a, b:
        for i in range(wp1, wp2+1):
            dist += dist(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
