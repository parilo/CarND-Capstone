#!/usr/bin/env python

import math
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLight

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
MANUVERS_ACCEL = 1.0 # maximum acceleration during aceeleration and deceleration manuvers
LAG_STEPS = 2 # number of teps which passes during calculations

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

        self.stop_waypoint_index = -1
        self.traffic_light_state = TrafficLight.RED
        self.current_velocity = 0.0
        self.max_velocity = rospy.get_param('/waypoint_loader/velocity') * 0.27778

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.base_waypoints = None
        self.current_waypoints_index = 0

        self.pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb, queue_size=1)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        rospy.Subscriber('/traffic_state', Int32, self.traffic_state_cb, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):

        # need to determine nearest waypoint and store it index
        # we assume that we going through waypoints continousely
        # and waypoints don't change, so we cache current waypoint and search
        # from current position onwards.

        if self.base_waypoints is None:
            return
        # return if we at the end of the path
        if self.current_waypoints_index == len(self.base_waypoints.waypoints) - 1:
            # reset to begining of the track
            self.current_waypoints_index = 0
            return

        car_pos =  msg.pose.position
        car_pos.z = 0
        # find the closest waypoint to current position
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

        if self.stop_waypoint_index != -1:
            waypoints_to_stop = self.stop_waypoint_index - self.current_waypoints_index
            if (
                (
                    self.traffic_light_state == TrafficLight.GREEN and
                    waypoints_to_stop < 10
                ) or
                waypoints_to_stop < 0
            ):
                waypoints_to_publish = LOOKAHEAD_WPS
            else:
                waypoints_to_publish = min(LOOKAHEAD_WPS, waypoints_to_stop)
        else:
            waypoints_to_publish = LOOKAHEAD_WPS

        if waypoints_to_publish > 1:
            waypoints_to_publish -= 1

        # print('--- index: {} {} {} {}'.format(self.current_waypoints_index, self.stop_waypoint_index, waypoints_to_stop, waypoints_to_publish))

        # get next LOOKAHEAD_WPS waypoints from current waypoint
        final_waypoints = Lane()
        final_waypoints.header.frame_id = '/world'
        final_waypoints.header.stamp = rospy.Time(0)
        final_waypoints_end_index = min(self.current_waypoints_index + waypoints_to_publish + 1, len(self.base_waypoints.waypoints))
        final_waypoints.waypoints = self.base_waypoints.waypoints[self.current_waypoints_index + 2:final_waypoints_end_index]

        self.plan_accel (final_waypoints)
        # plan to stop if it is not end of the track
        if self.stop_waypoint_index != -1:
            self.plan_slow_down (final_waypoints)

        self.final_waypoints_pub.publish(final_waypoints)

    def waypoints_cb(self, waypoints):
        # possibly we need to handle situation of waypoints chaning
        # and reset self.current_waypoints_index to 0 in this case
        # here
        self.base_waypoints = waypoints
        self.base_waypoints_sub.unregister()
        self.stop_waypoint_index = -1

    def traffic_cb(self, msg):
        # receiving waypoint index of traffic light stop line
        # or -1 otherwise
        if msg.data == -1:
            self.stop_waypoint_index = -1
        else:
            self.stop_waypoint_index = msg.data
        pass

    def traffic_state_cb(self, msg):
        # receiving traffic light state
        self.traffic_light_state = msg.data

    def obstacle_cb(self, msg):
        pass

    def velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

    def plan_accel(self, waypoints):
        '''
            Plan velocities of begining waypoints to make acceleration
            up to self.max_velocity

            Args:
                waypoints (styx_msgs/Lane): waypoints will be midified in place
        '''
        wp_count = len(waypoints.waypoints)
        if wp_count == 0:
            return

        dv = math.fabs(float(self.max_velocity - self.current_velocity))

        if wp_count < 2:
            return

        wp_length = self.distance_along_waypoints(waypoints.waypoints, 0, wp_count - 1)
        accel_time = dv / MANUVERS_ACCEL
        accel_length = self.current_velocity * accel_time + MANUVERS_ACCEL * accel_time * accel_time * 0.5
        accel_count = int(float(wp_count) * accel_length / wp_length)

        if accel_count != 0:
            time_for_step = accel_time / accel_count
        else:
            time_for_step = 0
        v = self.current_velocity
        v += LAG_STEPS * MANUVERS_ACCEL * time_for_step
        for i in range(wp_count):
            v += MANUVERS_ACCEL * time_for_step
            self.set_waypoint_velocity(waypoints.waypoints, i, min(v, self.max_velocity))

    def plan_slow_down(self, waypoints):
        '''
            Plan velocities of ending waypoints to make deceleration
            to 0 m/s

            Args:
                waypoints (styx_msgs/Lane): waypoints will be midified in place
        '''
        wp_count = len(waypoints.waypoints)
        if wp_count == 0:
            return

        path_velocity = self.get_waypoint_velocity(waypoints.waypoints[-1])
        dv = path_velocity

        if wp_count < 2:
            self.set_waypoint_velocity(waypoints.waypoints, 0, 0)
            return

        wp_length = self.distance_along_waypoints(waypoints.waypoints, 0, wp_count - 1)
        accel_time = dv / MANUVERS_ACCEL
        accel_length = self.current_velocity * accel_time + MANUVERS_ACCEL * accel_time * accel_time * 0.5
        accel_count = int(float(wp_count) * accel_length / wp_length)

        if accel_count != 0:
            time_for_step = accel_time / accel_count
        else:
            time_for_step = 0
        v = 0.0
        for i in range(wp_count):
            v += MANUVERS_ACCEL * time_for_step
            wp_index = wp_count - i - 1
            wp_vel = self.get_waypoint_velocity(waypoints.waypoints[wp_index])
            if wp_vel > v:
                self.set_waypoint_velocity(waypoints.waypoints, wp_index, v)

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
        distance = 0.0
        # dl = lambda a, b:
        for i in range(wp1, wp2+1):
            distance += dist(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return distance


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
