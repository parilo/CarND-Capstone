#!/usr/bin/python
# -*- coding: utf-8 -*-

import tf
import cv2
import yaml
import math
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
#   from traffic_light_config import config

# Our Team!
team = \
    """
          _, ___ __, __, __, , _   ___ _,_ __,    _, _
         (_   |  |_  |_  |_) \ |    |  |_| |_    / \ |
         , )  |  |   |   | \  \|    |  | | |     |~| |
          ~   ~  ~~~ ~~~ ~ ~   )    ~  ~ ~ ~~~   ~ ~ ~
                              ~'                      
        ================================================
    """

STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and 
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''

        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def euclidean_distance(self, x1, y1, x2, y2):
        """Calculate the distance between two coordinates modified version of waypoint_lodaer.py.
        http://www.mathwarehouse.com/algebra/distance_formula/index.php

        Args:
            x1 (double): x coordinate of the first point
            y1 (double): y coordinate of the first point
            x2 (double): x coordinate of the second point
            y2 (double): y coordinate of the second point

        Returns:
            Euclidean distance (double): Distance between two coordinates
        """
        x = x2 - x1
        y = y2 - y1
        return math.sqrt((x*x) + (y*y))

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
            https://www.cut-the-knot.org/pythagoras/DistanceFormula.shtml
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        if self.waypoints is not None:
            closest_dist = float('inf')
            closest_idx = 0

            for i, waypoint in enumerate(self.waypoints.waypoints):
                dist = self.euclidean_distance(pose.position.x, pose.position.y,
                                               waypoint.pose.pose.position.x, waypoint.pose.pose.position.y)
                if dist < closest_dist:
                    closest_idx = i
                    closest_dist = dist
                    #rospy.loginfo('Index {}'.format(closest_idx))
                    #rospy.loginfo('Distance {}'.format(closest_dist))

            return closest_idx

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # Principal x and y point of the image(Image Center)
        cx = image_width/2.0
        cy = image_height/2.0

        # get transform between pose of camera and world frame
        trans = None
        rot = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        if (trans is None) or (rot is None):
            return -1, -1

        # https://stackoverflow.com/questions/43849117/projecting-3d-model-onto-2d-plane
        # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

        # Creating 3D world point with numpy array
        object_point = np.array([[point_in_world.x, point_in_world.y, point_in_world.z]])

        # Transform quaternion to euler rotation matrix
        rot_mat = tf.transformations.euler_from_quaternion(rot)

        # Rotation vector and Translation vector
        rvec = np.array(rot_mat)
        tvec = np.array(trans)

        # create the camera matrix from the focal lengths and principal point
        camera_mat = np.matrix([[fx,  0, cx],
                                [ 0, fy, cy],
                                [ 0,  0,  1]])

        # Distortion coefficients
        distCoeffs = None

        # corresponding point in image from 3D world point
        img_point, jacobian = cv2.projectPoints(object_point, rvec, tvec, camera_mat, distCoeffs)

        # NOTE: THIS PART HAVEN'T FINISH YET

        return x, y

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            self.prev_light_loc = None
            return False

        rospy.loginfo("TEST")
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(light.pose.pose.position)

        # TODO use light location to zoom in on traffic light in image
        # <---- NOTE: THIS PART HAVEN'T FINISH YET

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def get_closest_traffic_light_distance(self, pose):
        """Identifies the closest traffic light to the current vehicle position

        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: Distance of the closest traffic light to the current vehicle position

        """
        traffic_light_positions = self.config['light_positions']
        closest_dist = float('inf')

        for light_position in traffic_light_positions:
            dist = self.euclidean_distance(pose.position.x, pose.position.y,
                                           light_position[0], light_position[1])
            dist -= 5   # Correcting to the traffic light for vehicle to stop inside the line
            if dist < closest_dist:
                closest_dist = dist
        return closest_dist

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = False
        car_position = None

        if self.pose:
            car_position = self.get_closest_waypoint(self.pose.pose)

        if car_position:
            closest_light = self.get_closest_traffic_light_distance(self.pose.pose)

            rospy.loginfo(team)
            rospy.loginfo("Current vehicle position: {}".format(car_position))
            rospy.loginfo("Closest traffic light: {}".format(closest_light))
            if closest_light < 50:
                rospy.loginfo("Traffic Light Detected!")
                # light = True # <---- NOTE: THIS PART HAVEN'T FINISH YET

        # TODO find the closest visible traffic light (if one exists)
        if light:
            rospy.loginfo("LIGHT TRUE")
            state = self.get_light_state(light)
            return light_wp, state

        self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
