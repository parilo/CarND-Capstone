import cv2
import numpy as np

from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN

    def get_test_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        img = np.asarray(image) # convert cv::Mat image to numpy array

        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #low_red = np.array([30, 150, 50])
        #upp_red = np.array([255, 255, 180])
        low_red = np.array([179, 50, 128])
        upp_red = np.array([179, 255, 255])

        low_yel = np.array([25, 50, 128])
        upp_yel = np.array([25, 255, 255])

        low_grn = np.array([60, 50, 128])
        upp_grn = np.array([60, 255, 255])

        mask = cv2.inRange(hsv, low_red, upp_red)
        res = cv2.bitwise_and(img, img, mask=mask)

        #cv2.imshow('Image', npimage) <-- does not work inside docker container

        cv2.imwrite('./images/testing.jpg', res)

        print img.shape
        print img.size
        print img.dtype

        return TrafficLight.UNKNOWN # GREEN, YELLOW, RED
