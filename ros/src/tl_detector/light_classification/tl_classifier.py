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
        #return image
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

        # See: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
        #
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the upper and lower boundaries for HSV
        # values which define a masking range for the
        # color of interest

        # For red (lower band)
        low_red0 = np.array([0, 100, 100])
        upp_red0 = np.array([12, 255, 255])

        # For red (upper band)
        low_red1 = np.array([167, 100, 100])
        upp_red1 = np.array([179, 255, 255])

        # For yellow
        low_yel = np.array([10, 0, 100])
        upp_yel = np.array([32, 255, 255])

        # For green
        low_grn = np.array([35, 80, 80])
        upp_grn = np.array([90, 255, 255])

        # Define the region-of-interest (roi) upper-left
        # corner coordinates (y, x); the full roi will be
        # calculated as (y:y+65, x:x+45)
        #
        roi_red = np.array([20, 20])
        roi_yel = np.array([98, 20])
        roi_grn = np.array([175, 20])

        # Build convenience array for masking infomation
        #
        masks = np.array([
            [low_red0, upp_red0, roi_red],
            [low_red1, upp_red1, roi_red],
            [low_yel, upp_yel, roi_yel],
            [low_grn, upp_grn, roi_grn]
        ])

        # Zero out ratios for later comparison
        #
        ratios = np.array([0.0, 0.0, 0.0, 0.0])

        # Iterate thru each of the masks...
        #
        for i, mask in enumerate(masks):

            # Get upper-left corner of the region-of-interest for the mask
            #
            y = mask[2][0]
            x = mask[2][1]

            # Grab the region-of-interest of the image and HSV mask
            #
            img_roi = img[y:y+65, x:x+45]
            hsv_roi = hsv[y:y+65, x:x+45]

            # Get the set of pixels found in the HSV range
            #
            hsv_set = cv2.inRange(hsv_roi, mask[0], mask[1])

            # Calculate the percentage ratio of found (masked)
            # pixels to the total region-of-interest (roi) pixels
            #
            #             num of pixels in mask
            # ratio = ------------------------------
            #         # of roi pixels (single plane)
            #
            ratio = (cv2.countNonZero(hsv_set) / (img_roi.size / 3.0))

            # Store the ratio for later comparison
            #
            ratios[i] = np.round(ratio, 1)

            # Uncomment this code if you want to see what was found
            res = cv2.bitwise_and(img_roi, img_roi, mask=hsv_set)
            color = ["red0", "red1", "yellow", "green"][i]
            cv2.imwrite('./images/' + color + '-testing-roi.jpg', img_roi)
            cv2.imwrite('./images/' + color + '-testing-res.jpg', res)

        #######################################################################

        top = 0.0
        tlv = TrafficLight.YELLOW
        diff = 0.15 # Decimal percentage difference to consider valid

        # Traffic light color enums
        colors = np.array([
            TrafficLight.RED,
            TrafficLight.RED,
            TrafficLight.YELLOW,
            TrafficLight.GREEN])

        # Find the most-likely traffic light color based on
        # found ratios and ratio differences
        for i, light in enumerate(colors):
            if ratios[i] > top and ((ratios[i] - top) / (top + 0.000000001)) > diff:
                top = ratios[i]
                tlv = colors[i]

        # Return the found traffic light color value
        return tlv
