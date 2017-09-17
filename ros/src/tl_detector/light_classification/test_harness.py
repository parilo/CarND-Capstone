#!/usr/bin/env python
from tl_classifier import TLClassifier
import cv2

class TestHarness():
    def __init__(self, cv_image):
        self.light_classifier = TLClassifier()
        return get_light_state(cv_image)

    def get_light_state(self, cv_image):
        return self.light_classifier.get_classification(cv_image)

if __name__ == '__main__':
    print TestHarness('image')
