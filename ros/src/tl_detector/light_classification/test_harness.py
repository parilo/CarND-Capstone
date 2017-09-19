#!/usr/bin/env python
from tl_classifier import TLClassifier
import cv2

class TestHarness():
    def __init__(self):
        self.light_classifier = TLClassifier()

    def get_light_state(self, cv_image):
        return self.light_classifier.get_test_classification(cv_image)

if __name__ == '__main__':
    test = TestHarness()
    npimage = cv2.imread('./images/red-cg-01.jpg')
    #npimage = cv2.imread('./images/yel-cg-01.jpg')
    #npimage = cv2.imread('./images/grn-cg-01.jpg') # not working?
    #matimage = cv2.fromarray(npimage) <-- does not work
    print test.get_light_state(npimage)
