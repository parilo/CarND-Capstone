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
    #color = 'red'
    color = 'yel'
    #color = 'grn'
    npimage = cv2.imread('./images/' + color + '-cg-01.jpg')

    print 'Testing for:', color
    print 'Found:', test.get_light_state(npimage)
